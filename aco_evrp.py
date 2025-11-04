#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACO + HGS-CVRP-style Enhancements for EVRP
------------------------------------------
This is a drop-in replacement of the original ACO EVRP solver, augmented with core ideas from
HGS-CVRP: population & elite pool, biased fitness (objective rank + diversity rank),
broken-pairs distance diversity control, local-search education (2-opt, relocate, swap),
elite-based pheromone reinforcement, and a light Split-like reconstruction from a giant tour.

Notes:
- Diversity measure ignores depot and stations, and only considers customer adjacency.
- Education operates on penalized objective (feasible solutions naturally favored).
- Split-like reconstruction is a greedy feasibility-aware segmenter for EVRP
  (not the exact DP Split of HGS-CVRP, but works well in practice for EVRP with stations).

CLI adds:
  --hgs-pool-size, --hgs-elite-size, --hgs-min-bpd, --education-iter,
  --education-max-neighbors, --enable-split-repair/--no-enable-split-repair
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


class ProgressPrinter:
    """Minimal terminal progress bar."""

    def __init__(self, total: int, label: str = "Progress", width: int = 30) -> None:
        self.total = max(1, total)
        self.label = label
        self.width = max(10, width)
        self.last_line_len = 0
        self.active = False

    def start(self) -> None:
        self.active = True
        self.update(0)

    def update(self, current: int) -> None:
        if not self.active:
            return
        fraction = min(max(current / self.total, 0.0), 1.0)
        filled = int(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        text = f"{self.label}: [{bar}] {current}/{self.total}"
        padding = " " * max(0, self.last_line_len - len(text))
        sys.stdout.write("\r" + text + padding)
        sys.stdout.flush()
        self.last_line_len = len(text)

    def close(self) -> None:
        if not self.active:
            return
        sys.stdout.write("\n")
        sys.stdout.flush()
        self.active = False

# =========================
# Problem & Solution Models
# =========================

@dataclass
class Solution:
    routes: List[List[int]]
    cost: float
    penalty: float
    unserved: List[int] = field(default_factory=list)
    feasible: bool = False
    energy_violation: float = 0.0
    time_violation: float = 0.0
    capacity_violation: float = 0.0

    @property
    def objective(self) -> float:
        return self.cost + self.penalty


@dataclass
class EVRPInstance:
    name: str
    num_vehicles: int
    capacity: int
    energy_capacity: float
    depot: int
    customers: List[int]
    stations: List[int]
    demands: List[int]
    energy: List[List[float]]
    travel_time: List[List[float]]
    heuristic: List[List[float]]
    min_energy_to_refill: List[float]
    max_route_time: float = 480.0
    customer_service_time: float = 20.0
    station_service_time: float = 60.0

    penalty_unserved: float = 10_000.0
    penalty_capacity: float = 5_000.0
    penalty_energy: float = 5_000.0
    penalty_time: float = 5_000.0

    def evaluate(self, routes: Sequence[Sequence[int]], unserved_hint: Optional[Iterable[int]] = None) -> Solution:
        """Compute total energy consumption and constraint violations for a solution."""
        visited = set()
        total_cost = 0.0
        penalty = 0.0
        energy_violation = 0.0
        time_violation = 0.0
        capacity_violation = 0.0

        for route in routes:
            if not route:
                penalty += self.penalty_energy
                continue

            if route[0] != self.depot:
                penalty += self.penalty_energy
            load = self.capacity
            energy = self.energy_capacity
            current = route[0]
            time_spent = 0.0

            for nxt in route[1:]:
                energy_required = self.energy[current][nxt]
                travel = self.travel_time[current][nxt]
                total_cost += energy_required
                time_spent += travel

                if energy_required > energy + 1e-9:
                    excess_energy = energy_required - energy
                    penalty += self.penalty_energy * excess_energy
                    energy_violation += excess_energy
                    energy = 0.0
                else:
                    energy -= energy_required
                energy = max(0.0, energy)

                if nxt == self.depot:
                    load = self.capacity
                    energy = self.energy_capacity
                elif nxt in self.stations:
                    energy = self.energy_capacity
                    time_spent += self.station_service_time
                else:
                    load -= self.demands[nxt]
                    if load < -1e-9:
                        excess_load = -load
                        penalty += self.penalty_capacity * excess_load
                        capacity_violation += excess_load
                        load = 0.0
                    visited.add(nxt)
                    time_spent += self.customer_service_time

                current = nxt

            if route[-1] != self.depot:
                penalty += self.penalty_energy
            if time_spent > self.max_route_time + 1e-6:
                excess_time = time_spent - self.max_route_time
                penalty += self.penalty_time * excess_time
                time_violation += excess_time

        if unserved_hint is not None:
            unserved = list(unserved_hint)
        else:
            unserved = [c for c in self.customers if c not in visited]
        if unserved:
            penalty += self.penalty_unserved * len(unserved)

        feasible = (
            penalty <= 1e-6
            and energy_violation <= 1e-6
            and time_violation <= 1e-6
            and capacity_violation <= 1e-6
            and not unserved
        )
        return Solution(
            routes=[list(r) for r in routes],
            cost=total_cost,
            penalty=penalty,
            unserved=unserved,
            feasible=feasible,
            energy_violation=energy_violation,
            time_violation=time_violation,
            capacity_violation=capacity_violation,
        )

    def route_feasible(self, route: Sequence[int]) -> bool:
        """Check whether a single route is feasible w.r.t. energy and capacity."""
        if not route or route[0] != self.depot or route[-1] != self.depot:
            return False

        load = self.capacity
        energy = self.energy_capacity
        current = route[0]
        time_spent = 0.0

        for nxt in route[1:]:
            required = self.energy[current][nxt]
            travel = self.travel_time[current][nxt]
            if required > energy + 1e-9:
                return False
            energy -= required
            time_spent += travel

            if nxt == self.depot:
                load = self.capacity
                energy = self.energy_capacity
            elif nxt in self.stations:
                energy = self.energy_capacity
                time_spent += self.station_service_time
            else:
                load -= self.demands[nxt]
                if load < -1e-9:
                    return False
                time_spent += self.customer_service_time

            if time_spent > self.max_route_time + 1e-9:
                return False

            current = nxt

        return True

    def service_time(self, node: int) -> float:
        if node == self.depot:
            return 0.0
        if node in self.stations:
            return self.station_service_time
        return self.customer_service_time


# =========================
# HGS-style Population
# =========================

def _solution_edges_customers_only(routes: Sequence[Sequence[int]], customers: set[int]) -> set[tuple[int, int]]:
    """Undirected edges between consecutive *customers* (ignore depot and stations)."""
    edges = set()
    for r in routes:
        prev_c = None
        for v in r:
            if v in customers:
                if prev_c is not None:
                    a, b = prev_c, v
                    if a > b:
                        a, b = b, a
                    edges.add((a, b))
                prev_c = v
            else:
                prev_c = prev_c  # skip non-customer (depot/station)
    return edges


def broken_pairs_distance(sol_a: Solution, sol_b: Solution, customers: set[int]) -> float:
    """Symmetric BPD between two solutions (only customer adjacencies)."""
    Ea = _solution_edges_customers_only(sol_a.routes, customers)
    Eb = _solution_edges_customers_only(sol_b.routes, customers)
    if not Ea and not Eb:
        return 0.0
    # BPD = 1 - (|Ea ∩ Eb| / |Ea ∪ Eb|)
    inter = len(Ea & Eb)
    union = len(Ea | Eb)
    if union == 0:
        return 0.0
    return 1.0 - (inter / union)


@dataclass
class PopMember:
    sol: Solution
    obj_rank: int = 0
    div_rank: int = 0
    biased_fitness: float = 0.0


class HGSPopulation:
    """Minimal HGS-style population with biased fitness and diversity control."""

    def __init__(self, customers: List[int], pool_size: int = 30, elite_size: int = 6, min_bpd: float = 0.15) -> None:
        self.customers_set = set(customers)
        self.pool_size = max(6, pool_size)
        self.elite_size = max(2, min(elite_size, self.pool_size // 3))
        self.min_bpd = min_bpd
        self.members: List[PopMember] = []

    def _recompute_biased_fitness(self) -> None:
        if not self.members:
            return
        n = len(self.members)
        # Objective ranks (ascending)
        order_obj = sorted(range(n), key=lambda i: self.members[i].sol.objective)
        for r, idx in enumerate(order_obj):
            self.members[idx].obj_rank = r + 1
        # Diversity: distance to closest neighbor (descending rank is better)
        distances = []
        for i in range(n):
            best = float("inf")
            for j in range(n):
                if i == j:
                    continue
                d = broken_pairs_distance(self.members[i].sol, self.members[j].sol, self.customers_set)
                if d < best:
                    best = d
            if best == float("inf"):
                best = 1.0
            distances.append(best)
        order_div = sorted(range(n), key=lambda i: -distances[i])
        for r, idx in enumerate(order_div):
            self.members[idx].div_rank = r + 1
        # Biased fitness = average of normalized ranks
        for m in self.members:
            m.biased_fitness = 0.5 * (m.obj_rank / n + m.div_rank / n)

    def _too_close_to_pool(self, sol: Solution) -> bool:
        for m in self.members:
            if broken_pairs_distance(sol, m.sol, self.customers_set) < self.min_bpd and m.sol.objective <= sol.objective + 1e-9:
                return True
        return False

    def try_add(self, sol: Solution) -> bool:
        """Add if diverse enough; maintain pool size via worst biased fitness eviction."""
        if self._too_close_to_pool(sol):
            return False
        self.members.append(PopMember(sol=sol))
        self._recompute_biased_fitness()
        if len(self.members) > self.pool_size:
            # Evict worst by biased fitness
            worst_idx = max(range(len(self.members)), key=lambda i: self.members[i].biased_fitness)
            self.members.pop(worst_idx)
            self._recompute_biased_fitness()
        return True

    def elite(self) -> List[Solution]:
        if not self.members:
            return []
        # Elite by objective (strict)
        ranked = sorted(self.members, key=lambda m: m.sol.objective)
        return [m.sol for m in ranked[: self.elite_size]]

    def best(self) -> Optional[Solution]:
        if not self.members:
            return None
        return min((m.sol for m in self.members), key=lambda s: s.objective)


# =========================
# Local Search (Education)
# =========================

class LSEngine:
    """Greedy improvement using relocate / swap / 2-opt within penalized objective."""

    def __init__(self, instance: EVRPInstance, rng: random.Random, max_neighbors: int = 50):
        self.inst = instance
        self.rng = rng
        self.max_neighbors = max_neighbors  # sampling cap per node to limit heavy evaluation

    def educate(self, sol: Solution, max_iter: int = 200) -> Solution:
        """Iterative best-improvement with a capped neighborhood enumeration."""
        current = sol
        improved = True
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1

            # Intra-route 2-opt
            better = self._best_two_opt(current)
            if better and better.objective + 1e-9 < current.objective:
                current = better
                improved = True
                continue

            # Inter/intra relocate
            better = self._best_relocate(current)
            if better and better.objective + 1e-9 < current.objective:
                current = better
                improved = True
                continue

            # Inter swap
            better = self._best_swap(current)
            if better and better.objective + 1e-9 < current.objective:
                current = better
                improved = True
                continue

        return current

    def _rebuild(self, routes: List[List[int]]) -> Solution:
        return self.inst.evaluate(routes)

    def _nodes_of_route(self, r: List[int]) -> List[int]:
        # Return indices of customer positions (exclude depot/stations)
        idxs = []
        for i, v in enumerate(r):
            if v != self.inst.depot and v not in self.inst.stations:
                idxs.append(i)
        return idxs

    def _best_two_opt(self, sol: Solution) -> Optional[Solution]:
        inst = self.inst
        best = None
        routes = [list(r) for r in sol.routes]
        for ridx, r in enumerate(routes):
            n = len(r)
            if n < 5:
                continue
            cust_pos = self._nodes_of_route(r)
            if len(cust_pos) < 3:
                continue
            # Sample pairs to limit time
            pairs_tested = 0
            for i_idx in range(len(cust_pos) - 1):
                for j_idx in range(i_idx + 1, len(cust_pos)):
                    if pairs_tested > self.max_neighbors:
                        break
                    i, j = cust_pos[i_idx], cust_pos[j_idx]
                    if i + 1 >= j:
                        continue
                    new_r = r[:i+1] + list(reversed(r[i+1:j+1])) + r[j+1:]
                    new_routes = [list(rr) for rr in routes]
                    new_routes[ridx] = new_r
                    cand = inst.evaluate(new_routes)
                    if best is None or cand.objective < best.objective:
                        best = cand
                    pairs_tested += 1
                if pairs_tested > self.max_neighbors:
                    break
        return best

    def _best_relocate(self, sol: Solution) -> Optional[Solution]:
        inst = self.inst
        best = None
        base_routes = [list(r) for r in sol.routes]

        # Enumerate a limited subset in random order
        route_indices = list(range(len(base_routes)))
        self.rng.shuffle(route_indices)

        for src_idx in route_indices:
            src = base_routes[src_idx]
            src_positions = self._nodes_of_route(src)
            if not src_positions:
                continue
            self.rng.shuffle(src_positions)
            for pos in src_positions[: max(1, len(src_positions))]:
                node = src[pos]
                for dst_idx in route_indices:
                    for ins_pos in range(1, len(base_routes[dst_idx])):  # after depot
                        if self.max_neighbors <= 0:
                            break
                        if src_idx == dst_idx and (ins_pos == pos or ins_pos == pos + 1):
                            continue
                        routes = [list(r) for r in base_routes]
                        # remove
                        routes[src_idx] = routes[src_idx][:pos] + routes[src_idx][pos+1:]
                        # ensure route not broken (keep depot at end)
                        if len(routes[src_idx]) == 0 or routes[src_idx][0] != inst.depot:
                            routes[src_idx] = [inst.depot] + routes[src_idx]
                        if routes[src_idx][-1] != inst.depot:
                            routes[src_idx].append(inst.depot)
                        # insert
                        if ins_pos >= len(routes[dst_idx]):
                            ins_pos = len(routes[dst_idx]) - 1
                        routes[dst_idx] = routes[dst_idx][:ins_pos] + [node] + routes[dst_idx][ins_pos:]
                        cand = inst.evaluate(routes)
                        if best is None or cand.objective < best.objective:
                            best = cand
                        self.max_neighbors -= 1
                        if self.max_neighbors <= 0:
                            return best
        return best

    def _best_swap(self, sol: Solution) -> Optional[Solution]:
        inst = self.inst
        best = None
        routes = [list(r) for r in sol.routes]

        idxs = list(range(len(routes)))
        self.rng.shuffle(idxs)
        for i in idxs:
            for j in idxs:
                if j <= i:
                    continue
                r1, r2 = routes[i], routes[j]
                pos1 = self._nodes_of_route(r1)
                pos2 = self._nodes_of_route(r2)
                if not pos1 or not pos2:
                    continue
                self.rng.shuffle(pos1)
                self.rng.shuffle(pos2)
                for a in pos1[: min(len(pos1), 5)]:
                    for b in pos2[: min(len(pos2), 5)]:
                        if self.max_neighbors <= 0:
                            return best
                        v1, v2 = r1[a], r2[b]
                        nr1 = r1[:a] + [v2] + r1[a+1:]
                        nr2 = r2[:b] + [v1] + r2[b+1:]
                        nroutes = [list(r) for r in routes]
                        nroutes[i] = nr1
                        nroutes[j] = nr2
                        cand = inst.evaluate(nroutes)
                        if best is None or cand.objective < best.objective:
                            best = cand
                        self.max_neighbors -= 1
        return best


class SwapStarEngine:
    """SWAP* multi-neighborhood improver (swap, N1, N2, intra-route, inter-route, LNS)."""

    def __init__(
        self,
        instance: EVRPInstance,
        rng: random.Random,
        *,
        max_iterations: int = 4,
        sample_size: int = 60,
        lns_removal_rate: float = 0.2,
        lns_min_remove: int = 2,
    ) -> None:
        self.inst = instance
        self.rng = rng
        self.max_iterations = max(1, max_iterations)
        self.sample_size = max(1, sample_size)
        self.lns_removal_rate = min(max(0.0, lns_removal_rate), 0.8)
        self.lns_min_remove = max(1, lns_min_remove)

    def improve(self, solution: Solution) -> Solution:
        current = solution
        for _ in range(self.max_iterations):
            improved = False
            for move in (
                self._swap_nodes,
                self._n1_random_insert,
                self._n2_swap_reinsert,
                self._intra_route_reorder,
                self._inter_route_exchange,
                self._large_neighborhood_search,
            ):
                candidate = move(current)
                if candidate and candidate.objective + 1e-9 < current.objective:
                    current = candidate
                    improved = True
            if not improved:
                break
        return current

    def _is_customer(self, node: int) -> bool:
        if node == self.inst.depot or node in self.inst.stations:
            return False
        return True

    def _customer_positions(self, route: List[int]) -> List[int]:
        return [
            idx
            for idx, node in enumerate(route)
            if idx not in (0, len(route) - 1) and self._is_customer(node)
        ]

    def _copy_routes(self, routes: Sequence[Sequence[int]]) -> List[List[int]]:
        return [list(r) for r in routes]

    def _ensure_route_shape(self, route: List[int]) -> None:
        depot = self.inst.depot
        if not route:
            route.extend([depot, depot])
            return
        if route[0] != depot:
            route.insert(0, depot)
        if route[-1] != depot:
            route.append(depot)
        if len(route) == 1:
            route.append(depot)

    # --- Neighborhoods ---

    def _swap_nodes(self, sol: Solution) -> Optional[Solution]:
        routes = self._copy_routes(sol.routes)
        inst = self.inst
        best: Optional[Solution] = None
        attempts = 0
        attempt_cap = self.sample_size
        idxs = list(range(len(routes)))
        self.rng.shuffle(idxs)
        for i_idx, ridx in enumerate(idxs):
            r1 = routes[ridx]
            pos1 = self._customer_positions(r1)
            if not pos1:
                continue
            for jidx in idxs[i_idx + 1 :]:
                r2 = routes[jidx]
                pos2 = self._customer_positions(r2)
                if not pos2:
                    continue
                sample_pos1 = pos1 if len(pos1) <= 5 else self.rng.sample(pos1, 5)
                sample_pos2 = pos2 if len(pos2) <= 5 else self.rng.sample(pos2, 5)
                for a in sample_pos1:
                    for b in sample_pos2:
                        new_routes = self._copy_routes(routes)
                        new_routes[ridx][a], new_routes[jidx][b] = new_routes[jidx][b], new_routes[ridx][a]
                        cand = inst.evaluate(new_routes)
                        if best is None or cand.objective < best.objective - 1e-9:
                            best = cand
                        attempts += 1
                        if attempts >= attempt_cap:
                            return best
        return best

    def _n1_random_insert(self, sol: Solution) -> Optional[Solution]:
        """N1: pick a random node and try inserting it in other sub-routes."""
        inst = self.inst
        routes = self._copy_routes(sol.routes)
        candidates: List[Tuple[int, int]] = []
        for ridx, route in enumerate(routes):
            for pos in self._customer_positions(route):
                candidates.append((ridx, pos))
        if len(candidates) <= 1 or len(routes) <= 1:
            return None
        self.rng.shuffle(candidates)
        best: Optional[Solution] = None
        attempt_cap = self.sample_size
        attempts = 0

        for ridx, pos in candidates:
            if attempts >= attempt_cap:
                break
            base_routes = self._copy_routes(routes)
            node = base_routes[ridx].pop(pos)
            self._ensure_route_shape(base_routes[ridx])
            destinations = [idx for idx in range(len(routes)) if idx != ridx]
            self.rng.shuffle(destinations)
            for dst_idx in destinations:
                dst_route = base_routes[dst_idx]
                if len(dst_route) < 1:
                    continue
                for insert_at in range(1, len(dst_route)):
                    attempts += 1
                    trial_routes = self._copy_routes(base_routes)
                    trial_routes[dst_idx].insert(insert_at, node)
                    self._ensure_route_shape(trial_routes[dst_idx])
                    cand = inst.evaluate(trial_routes)
                    if best is None or cand.objective < best.objective - 1e-9:
                        best = cand
                    if attempts >= attempt_cap:
                        return best
        return best

    def _n2_swap_reinsert(self, sol: Solution) -> Optional[Solution]:
        """N2: swap two nodes and reinsert each into the best position."""
        inst = self.inst
        routes = self._copy_routes(sol.routes)
        customer_refs: List[Tuple[int, int, int]] = []
        for ridx, route in enumerate(routes):
            for pos in self._customer_positions(route):
                customer_refs.append((ridx, pos, route[pos]))
        if len(customer_refs) < 2:
            return None
        self.rng.shuffle(customer_refs)
        best: Optional[Solution] = None
        attempt_cap = self.sample_size
        attempts = 0

        for idx1 in range(len(customer_refs) - 1):
            ridx1, pos1, node1 = customer_refs[idx1]
            for idx2 in range(idx1 + 1, len(customer_refs)):
                ridx2, pos2, node2 = customer_refs[idx2]
                base_routes = self._copy_routes(routes)
                removals = sorted(
                    [(ridx1, pos1, node1), (ridx2, pos2, node2)],
                    key=lambda x: (x[0], x[1]),
                    reverse=True,
                )
                for r_idx, r_pos, expected_node in removals:
                    if r_pos >= len(base_routes[r_idx]):
                        break
                    removed = base_routes[r_idx].pop(r_pos)
                    if removed != expected_node:
                        # Node mismatch indicates duplicate values or prior removal; skip this pair.
                        break
                else:
                    touched = {ridx1, ridx2}
                    for ridx in touched:
                        self._ensure_route_shape(base_routes[ridx])

                    target_a = ridx2
                    target_b = ridx1
                    route_a = base_routes[target_a]
                    route_b = base_routes[target_b]
                    if len(route_a) < 2 or len(route_b) < 2:
                        continue
                    for insert_a in range(1, len(route_a)):
                        temp_routes = self._copy_routes(base_routes)
                        temp_routes[target_a].insert(insert_a, node1)
                        self._ensure_route_shape(temp_routes[target_a])
                        for insert_b in range(1, len(temp_routes[target_b])):
                            attempts += 1
                            trial_routes = self._copy_routes(temp_routes)
                            trial_routes[target_b].insert(insert_b, node2)
                            self._ensure_route_shape(trial_routes[target_b])
                            cand = inst.evaluate(trial_routes)
                            if best is None or cand.objective < best.objective - 1e-9:
                                best = cand
                            if attempts >= attempt_cap:
                                return best
                if attempts >= attempt_cap:
                    return best
        return best

    def _intra_route_reorder(self, sol: Solution) -> Optional[Solution]:
        routes = self._copy_routes(sol.routes)
        inst = self.inst
        best: Optional[Solution] = None
        attempts = 0
        attempt_cap = self.sample_size
        for ridx, route in enumerate(routes):
            pos = self._customer_positions(route)
            if len(pos) < 2:
                continue
            max_seg = min(3, len(pos))
            for start_idx in range(len(pos)):
                for seg_len in range(1, max_seg + 1):
                    if start_idx + seg_len > len(pos):
                        break
                    start = pos[start_idx]
                    end = pos[start_idx + seg_len - 1]
                    segment = route[start : end + 1]
                    base = route[:start] + route[end + 1 :]
                    if len(base) < 2:
                        continue
                    for insert_at in range(1, len(base)):
                        attempts += 1
                        if attempts > attempt_cap:
                            return best
                        new_route = base[:insert_at] + segment + base[insert_at:]
                        self._ensure_route_shape(new_route)
                        new_routes = self._copy_routes(routes)
                        new_routes[ridx] = new_route
                        cand = inst.evaluate(new_routes)
                        if best is None or cand.objective < best.objective - 1e-9:
                            best = cand
        return best

    def _inter_route_exchange(self, sol: Solution) -> Optional[Solution]:
        routes = self._copy_routes(sol.routes)
        inst = self.inst
        best: Optional[Solution] = None
        attempts = 0
        attempt_cap = self.sample_size
        idxs = list(range(len(routes)))
        self.rng.shuffle(idxs)
        for src_idx in idxs:
            source = routes[src_idx]
            src_pos = self._customer_positions(source)
            if not src_pos:
                continue
            sample_src = src_pos if len(src_pos) <= 5 else self.rng.sample(src_pos, 5)
            for pos in sample_src:
                node = source[pos]
                for dst_idx in idxs:
                    if dst_idx == src_idx:
                        continue
                    target = routes[dst_idx]
                    for insert_at in range(1, len(target)):
                        attempts += 1
                        new_routes = self._copy_routes(routes)
                        moved = new_routes[src_idx].pop(pos)
                        self._ensure_route_shape(new_routes[src_idx])
                        new_routes[dst_idx].insert(insert_at, moved)
                        self._ensure_route_shape(new_routes[dst_idx])
                        cand = inst.evaluate(new_routes)
                        if best is None or cand.objective < best.objective - 1e-9:
                            best = cand
                        if attempts >= attempt_cap:
                            return best
        return best

    def _large_neighborhood_search(self, sol: Solution) -> Optional[Solution]:
        routes = self._copy_routes(sol.routes)
        inst = self.inst
        customer_refs: List[Tuple[int, int, int]] = []
        for ridx, route in enumerate(routes):
            for pos in self._customer_positions(route):
                customer_refs.append((ridx, pos, route[pos]))
        if not customer_refs:
            return None

        removal_count = max(
            self.lns_min_remove, int(len(customer_refs) * self.lns_removal_rate)
        )
        removal_count = min(removal_count, len(customer_refs))
        if removal_count <= 0:
            return None
        selected = self.rng.sample(customer_refs, removal_count)
        selected.sort(key=lambda x: (x[0], x[1]), reverse=True)
        removed_nodes: List[int] = []
        for ridx, pos, node in selected:
            removed_nodes.append(node)
            routes[ridx].pop(pos)
            self._ensure_route_shape(routes[ridx])

        if not removed_nodes:
            return None

        for node in removed_nodes:
            best_candidate: Optional[Solution] = None
            best_routes: Optional[List[List[int]]] = None
            for ridx, route in enumerate(routes):
                for insert_at in range(1, len(route)):
                    trial_routes = self._copy_routes(routes)
                    trial_routes[ridx].insert(insert_at, node)
                    cand = inst.evaluate(trial_routes)
                    if best_candidate is None or cand.objective < best_candidate.objective - 1e-9:
                        best_candidate = cand
                        best_routes = trial_routes
            if best_candidate is None or best_routes is None:
                routes[0].insert(len(routes[0]) - 1, node)
                self._ensure_route_shape(routes[0])
            else:
                routes = self._copy_routes(best_routes)

        final = inst.evaluate(routes)
        return final


# =========================
# ACO with HGS Enhancements
# =========================

class ACOEvrpSolver:
    """Ant Colony solver tuned for EVRP + HGS-CVRP style enhancements."""

    def __init__(
        self,
        instance: EVRPInstance,
        *,
        seed: Optional[int] = None,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        q0: float = 0.2,
        num_ants: Optional[int] = None,
        initial_pheromone: Optional[float] = None,
        time_relaxation: float = 0.15,
        penalty_update_interval: int = 25,
        # HGS options
        hgs_pool_size: int = 30,
        hgs_elite_size: int = 6,
        hgs_min_bpd: float = 0.15,
        education_iter: int = 200,
        education_max_neighbors: int = 200,
        enable_swap_star: bool = True,
        swap_star_iterations: int = 4,
        swap_star_sample_size: int = 60,
        swap_star_removal_rate: float = 0.2,
        swap_star_min_remove: int = 2,
        enable_split_repair: bool = True,
        show_progress: bool = True,
        progress_label: Optional[str] = None,
    ) -> None:
        self.instance = instance
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.num_ants = num_ants or max(10, len(instance.customers) // 3)
        self.initial_pheromone = initial_pheromone or self._default_initial_pheromone()
        self.time_relaxation = max(0.0, time_relaxation)
        self.penalty_update_interval = max(1, penalty_update_interval)
        self.show_progress = show_progress and sys.stdout.isatty()
        self.progress_label = progress_label or instance.name or "ACO"

        self.rng = random.Random(seed)
        self.pheromone = self._init_pheromone_matrix(self.initial_pheromone)
        self.global_best: Optional[Solution] = None
        self._feasible_counter = 0
        self._infeasible_counter = 0
        self._base_penalties = {
            "time": instance.penalty_time,
            "capacity": instance.penalty_capacity,
            "energy": instance.penalty_energy,
            "unserved": instance.penalty_unserved,
        }

        # HGS
        self.population = HGSPopulation(instance.customers, pool_size=hgs_pool_size, elite_size=hgs_elite_size, min_bpd=hgs_min_bpd)
        self.education_iter = max(0, education_iter)
        self.education_max_neighbors = max(10, education_max_neighbors)
        self.enable_swap_star = enable_swap_star and swap_star_iterations > 0
        self.swap_star_iterations = max(0, swap_star_iterations)
        self.swap_star_sample_size = max(1, swap_star_sample_size)
        self.swap_star_removal_rate = max(0.0, swap_star_removal_rate)
        self.swap_star_min_remove = max(1, swap_star_min_remove)
        self.enable_split_repair = enable_split_repair

    def _default_initial_pheromone(self) -> float:
        total_energy = sum(sum(row) for row in self.instance.energy)
        avg_energy = total_energy / (len(self.instance.energy) ** 2)
        return 1.0 / max(avg_energy, 1e-3)

    def _init_pheromone_matrix(self, value: float) -> List[List[float]]:
        size = len(self.instance.energy)
        return [[value if i != j else 0.0 for j in range(size)] for i in range(size)]

    def _record_feasibility(self, solution: Solution) -> None:
        if solution.feasible:
            self._feasible_counter += 1
        else:
            self._infeasible_counter += 1

    def _tune_penalties(self) -> None:
        total = self._feasible_counter + self._infeasible_counter
        if total < self.penalty_update_interval:
            return

        ratio = self._feasible_counter / max(1, total)
        target = 0.2
        inst = self.instance
        growth = 1.05
        decay = 0.95

        if ratio < target:
            inst.penalty_time *= growth
            inst.penalty_capacity *= growth
            inst.penalty_energy *= growth
            inst.penalty_unserved *= growth
        elif ratio > target * 1.5:
            inst.penalty_time = max(self._base_penalties["time"], inst.penalty_time * decay)
            inst.penalty_capacity = max(self._base_penalties["capacity"], inst.penalty_capacity * decay)
            inst.penalty_energy = max(self._base_penalties["energy"], inst.penalty_energy * decay)
            inst.penalty_unserved = max(self._base_penalties["unserved"], inst.penalty_unserved * decay)

        self._feasible_counter = 0
        self._infeasible_counter = 0

    def run(self, iterations: int) -> Solution:
        """Execute the main optimisation loop (ACO + education + population)."""
        ls_engine = LSEngine(self.instance, self.rng, max_neighbors=self.education_max_neighbors)
        swap_star_engine: Optional[SwapStarEngine] = None
        if self.enable_swap_star and self.swap_star_iterations > 0:
            swap_star_engine = SwapStarEngine(
                self.instance,
                self.rng,
                max_iterations=self.swap_star_iterations,
                sample_size=self.swap_star_sample_size,
                lns_removal_rate=self.swap_star_removal_rate,
                lns_min_remove=self.swap_star_min_remove,
            )

        # Warm start: insert dummy initial routes (each customer alone)
        init_routes = []
        for _ in range(self.instance.num_vehicles):
            init_routes.append([self.instance.depot, self.instance.depot])
        warm = self.instance.evaluate(init_routes, unserved_hint=self.instance.customers)
        self.population.try_add(warm)

        progress = None
        if self.show_progress and iterations > 0:
            progress = ProgressPrinter(iterations, self.progress_label, width=40)
            progress.start()

        final_best: Optional[Solution] = None
        try:
            for iteration in range(1, iterations + 1):
                iteration_best: Optional[Solution] = None

                for _ in range(self.num_ants):
                    candidate = self._construct_solution()

                    if self.enable_split_repair:
                        candidate = self._split_like_repair(candidate)

                    if swap_star_engine is not None:
                        candidate = swap_star_engine.improve(candidate)

                    if self.education_iter > 0:
                        candidate = ls_engine.educate(candidate, max_iter=self.education_iter)

                    improved = self._remove_redundant_stations(candidate)
                    if improved.objective < candidate.objective:
                        candidate = improved

                    if iteration_best is None or candidate.objective < iteration_best.objective:
                        iteration_best = candidate
                    if self.global_best is None or candidate.objective < self.global_best.objective:
                        self.global_best = candidate
                    self._record_feasibility(candidate)
                    self.population.try_add(candidate)

                if iteration_best is None:
                    if progress:
                        progress.update(iteration)
                    continue

                self._evaporate_pheromone()
                self._reinforce_pheromone(iteration_best, factor=0.25)
                elites = self.population.elite()
                if elites:
                    fac = 0.75 / len(elites)
                    for e in elites:
                        self._reinforce_pheromone(e, factor=fac)

                self._tune_penalties()
                if progress:
                    progress.update(iteration)

            final_best = self.population.best() or self.global_best
        finally:
            if progress:
                progress.close()

        if final_best is None:
            raise RuntimeError("Solver failed to produce any solution.")
        return final_best

    # ---------- Construction ----------

    def _construct_solution(self) -> Solution:
        inst = self.instance
        unserved = set(inst.customers)
        routes: List[List[int]] = []

        for _vehicle in range(inst.num_vehicles):
            if not unserved:
                routes.append([inst.depot, inst.depot])
                continue

            time_limit = inst.max_route_time * (1.0 + self.time_relaxation)
            route = [inst.depot]
            load = inst.capacity
            energy = inst.energy_capacity
            current = inst.depot
            route_time = 0.0
            stagnation_counter = 0

            while True:
                feasible_customers: List[int] = []
                attractiveness: List[float] = []

                for cust in unserved:
                    demand = inst.demands[cust]
                    if demand > load:
                        continue
                    energy_to_cust = inst.energy[current][cust]
                    if energy_to_cust > energy:
                        continue
                    travel_to_cust = inst.travel_time[current][cust]
                    service = inst.customer_service_time
                    new_time = route_time + travel_to_cust + service
                    if new_time > time_limit + 1e-9:
                        continue
                    if new_time + inst.travel_time[cust][inst.depot] > time_limit + 1e-9:
                        continue
                    remaining_energy = energy - energy_to_cust

                    min_after_visit = inst.min_energy_to_refill[cust]
                    depot_energy = inst.energy[cust][inst.depot]
                    if remaining_energy < min(min_after_visit, depot_energy) - 1e-9:
                        continue

                    feasible_customers.append(cust)
                    attractiveness.append(self._edge_attractiveness(current, cust))

                if feasible_customers:
                    next_customer = self._select_next_node(feasible_customers, attractiveness)
                    energy_cost = inst.energy[current][next_customer]
                    travel_time = inst.travel_time[current][next_customer]

                    route.append(next_customer)
                    load -= inst.demands[next_customer]
                    energy -= energy_cost
                    current = next_customer
                    route_time += travel_time + inst.customer_service_time
                    unserved.remove(next_customer)
                    stagnation_counter = 0

                    if not unserved:
                        break
                    continue

                demand_fit = any(inst.demands[c] <= load for c in unserved)
                if not demand_fit:
                    if current != inst.depot:
                        route.append(inst.depot)
                    break

                # Recharge or close route
                station_candidates, station_attractiveness = self._reachable_stations(
                    current,
                    energy,
                    route_time,
                    require_progress=bool(unserved),
                )

                if station_candidates:
                    chosen_station = self._select_next_node(station_candidates, station_attractiveness)
                    energy_to_station = inst.energy[current][chosen_station]
                    travel_to_station = inst.travel_time[current][chosen_station]
                    energy -= energy_to_station
                    current = chosen_station
                    energy = inst.energy_capacity
                    route.append(chosen_station)
                    route_time += travel_to_station + inst.station_service_time
                    if route_time > time_limit + 1e-9:
                        route_time = time_limit + 1e-6
                    stagnation_counter += 1

                    if stagnation_counter > len(inst.stations) + 2:
                        break
                    continue

                if current != inst.depot:
                    travel_back = inst.travel_time[current][inst.depot]
                    if route_time + travel_back <= time_limit + 1e-9:
                        route.append(inst.depot)
                    else:
                        route.append(inst.depot)
                break

            if route[-1] != inst.depot:
                if inst.energy[current][inst.depot] <= energy + 1e-9:
                    route.append(inst.depot)
                else:
                    route.append(inst.depot)

            routes.append(route)

        solution = inst.evaluate(routes, unserved_hint=sorted(unserved))
        return solution

    # ---------- HGS-inspired Split-like repair ----------

    def _split_like_repair(self, sol: Solution) -> Solution:
        """
        Build a giant tour (customers only) from the solution's visit order and greedily
        segment into feasible routes (capacity/energy/time via simulate). Then evaluate and keep if not worse.
        """
        inst = self.instance
        # Extract visit order by first appearance
        seen = set()
        order: List[int] = []
        for r in sol.routes:
            for v in r:
                if v in inst.customers and v not in seen:
                    seen.add(v)
                    order.append(v)
        if not order:
            return sol

        routes: List[List[int]] = []
        i = 0
        while i < len(order):
            r = [inst.depot]
            load = inst.capacity
            energy = inst.energy_capacity
            time_spent = 0.0
            cur = inst.depot

            j = i
            while j < len(order):
                nxt = order[j]
                dem = inst.demands[nxt]
                if dem > load:
                    break
                e_need = inst.energy[cur][nxt]
                t_need = inst.travel_time[cur][nxt] + inst.customer_service_time
                # Try visit then possibly recharge to return
                if e_need > energy + 1e-9:
                    # attempt recharge to nearest station that keeps time feasible
                    picked = None
                    for s in inst.stations:
                        if inst.energy[cur][s] <= energy + 1e-9:
                            if time_spent + inst.travel_time[cur][s] + inst.station_service_time <= inst.max_route_time * (1 + self.time_relaxation):
                                picked = s
                                break
                    if picked is None:
                        break
                    # go to station
                    time_spent += inst.travel_time[cur][picked] + inst.station_service_time
                    energy -= inst.energy[cur][picked]
                    energy = inst.energy_capacity
                    r.append(picked)
                    cur = picked
                    # re-check after recharge
                    if inst.energy[cur][nxt] > energy + 1e-9:
                        break
                    e_need = inst.energy[cur][nxt]
                    t_need = inst.travel_time[cur][nxt] + inst.customer_service_time

                # time feasibility (including ability to return to depot)
                if time_spent + t_need + inst.travel_time[nxt][inst.depot] > inst.max_route_time * (1 + self.time_relaxation) + 1e-9:
                    break

                # accept customer
                r.append(nxt)
                load -= dem
                time_spent += t_need
                energy -= e_need
                cur = nxt
                j += 1

            # close route
            if r[-1] != inst.depot:
                r.append(inst.depot)
            routes.append(r)
            if j == i:  # failed to add any -> avoid infinite loop by forcing one customer
                routes[-1] = [inst.depot, order[i], inst.depot]
                i = i + 1
            else:
                i = j

        cand = inst.evaluate(routes)
        return cand if cand.objective <= sol.objective + 1e-9 else sol

    # ---------- Post-processing ----------

    def _remove_redundant_stations(self, solution: Solution) -> Solution:
        inst = self.instance
        improved_routes: List[List[int]] = []

        for route in solution.routes:
            simplified = list(route)
            changed = True
            while changed:
                changed = False
                for idx in range(1, len(simplified) - 1):
                    node = simplified[idx]
                    if node not in inst.stations:
                        continue
                    trial_route = simplified[:idx] + simplified[idx + 1 :]
                    if inst.route_feasible(trial_route):
                        simplified = trial_route
                        changed = True
                        break
            improved_routes.append(simplified)

        improved_solution = inst.evaluate(improved_routes)
        if improved_solution.objective <= solution.objective:
            return improved_solution
        return solution

    # ---------- Helper: Reachable stations ----------

    def _reachable_stations(self, current: int, energy: float, route_time: float, require_progress: bool) -> Tuple[List[int], List[float]]:
        inst = self.instance
        time_limit = inst.max_route_time * (1.0 + self.time_relaxation)
        candidates: List[int] = []
        attractiveness: List[float] = []
        for station in inst.stations:
            if station == current:
                continue
            cost = inst.energy[current][station]
            if cost > energy + 1e-9:
                continue
            travel_time = inst.travel_time[current][station]
            service_time = inst.station_service_time
            new_time = route_time + travel_time + service_time
            if new_time > time_limit + 1e-9:
                continue
            if new_time + inst.travel_time[station][inst.depot] > time_limit + 1e-9:
                continue
            if require_progress and cost < 1e-9:
                continue
            candidates.append(station)
            attractiveness.append(self._edge_attractiveness(current, station))
        return candidates, attractiveness

    # ---------- ACO internals ----------

    def _edge_attractiveness(self, i: int, j: int) -> float:
        pheromone = max(self.pheromone[i][j], 1e-9)
        heuristic = max(self.instance.heuristic[i][j], 1e-9)
        return (pheromone**self.alpha) * (heuristic**self.beta)

    def _select_next_node(self, candidates: List[int], attractiveness: List[float]) -> int:
        if not candidates:
            raise ValueError("No candidates provided for selection.")
        if len(candidates) == 1 or not attractiveness:
            return candidates[0]

        q = self.rng.random()
        if q <= self.q0:
            best_index = max(range(len(candidates)), key=lambda idx: attractiveness[idx])
            return candidates[best_index]

        total = sum(attractiveness)
        if total <= 0.0:
            return self.rng.choice(candidates)

        threshold = self.rng.random() * total
        cumulative = 0.0
        for candidate, attr in zip(candidates, attractiveness):
            cumulative += attr
            if cumulative >= threshold:
                return candidate
        return candidates[-1]

    def _evaporate_pheromone(self) -> None:
        size = len(self.pheromone)
        for i in range(size):
            for j in range(size):
                if i == j:
                    continue
                self.pheromone[i][j] *= (1.0 - self.rho)
                if self.pheromone[i][j] < 1e-9:
                    self.pheromone[i][j] = 1e-9

    def _reinforce_pheromone(self, solution: Solution, factor: float) -> None:
        if solution.cost <= 0.0:
            return
        deposit = factor / solution.cost
        for route in solution.routes:
            for u, v in zip(route[:-1], route[1:]):
                if u == v:
                    continue
                self.pheromone[u][v] += deposit
                self.pheromone[v][u] += deposit


# =========================
# IO Utilities (unchanged)
# =========================

def load_matrix(path: Path) -> List[List[float]]:
    matrix: List[List[float]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            row = [float(value) for value in stripped.split()]
            matrix.append(row)

    if not matrix:
        raise ValueError(f"Matrix file {path} is empty.")

    width = len(matrix[0])
    for row in matrix:
        if len(row) != width:
            raise ValueError(f"Inconsistent row length in matrix file {path}.")

    return matrix


def build_heuristic_from_energy(energy: List[List[float]]) -> List[List[float]]:
    heuristic: List[List[float]] = []
    for row_idx, row in enumerate(energy):
        heuristic_row: List[float] = []
        for col_idx, value in enumerate(row):
            if row_idx == col_idx or value <= 0.0:
                heuristic_row.append(0.0)
            else:
                heuristic_row.append(1.0 / value)
        heuristic.append(heuristic_row)
    return heuristic


def load_instance(base_path: Path, heuristic_path: Optional[Path]) -> EVRPInstance:
    data_path = base_path.with_suffix(".txt")
    energy_path = base_path.with_name(base_path.name + "-energy.txt")
    time_path = base_path.with_name(base_path.name + "-time.txt")

    if not data_path.exists():
        raise FileNotFoundError(f"Instance description not found: {data_path}")
    if not energy_path.exists():
        raise FileNotFoundError(f"Energy matrix not found: {energy_path}")
    if not time_path.exists():
        raise FileNotFoundError(f"Travel time matrix not found: {time_path}")

    energy_matrix = load_matrix(energy_path)
    time_matrix = load_matrix(time_path)
    heuristic_matrix = load_matrix(heuristic_path) if heuristic_path else build_heuristic_from_energy(energy_matrix)

    if len(energy_matrix) != len(heuristic_matrix):
        raise ValueError("Energy and heuristic matrices must share identical dimensions.")
    if len(time_matrix) != len(energy_matrix):
        raise ValueError("Energy and time matrices must share identical dimensions.")

    metadata = _parse_instance_metadata(data_path)
    num_nodes = len(energy_matrix)

    demands = metadata["demands"]
    if len(demands) < num_nodes:
        demands.extend([0] * (num_nodes - len(demands)))
    elif len(demands) > num_nodes:
        demands = demands[:num_nodes]

    stations = [s for s in metadata["stations"] if 0 <= s < num_nodes]
    depot = metadata["depot"]

    customers = [idx for idx, demand in enumerate(demands) if idx != depot and demand > 0]

    min_energy_to_refill = []
    refill_targets = {depot, *stations}
    for idx in range(num_nodes):
        minimum = min(
            (
                energy_matrix[idx][target]
                for target in refill_targets
                if target < len(energy_matrix[idx])
                and energy_matrix[idx][target] <= metadata["energy_capacity"] + 1e-9
            ),
            default=float("inf"),
        )
        min_energy_to_refill.append(minimum if math.isfinite(minimum) else float("inf"))

    return EVRPInstance(
        name=metadata["name"],
        num_vehicles=metadata["vehicles"],
        capacity=metadata["capacity"],
        energy_capacity=metadata["energy_capacity"],
        depot=depot,
        customers=customers,
        stations=stations,
        demands=demands,
        energy=energy_matrix,
        travel_time=time_matrix,
        heuristic=heuristic_matrix,
        min_energy_to_refill=min_energy_to_refill,
    )


def _parse_instance_metadata(path: Path) -> dict:
    vehicles = capacity = 0
    energy_capacity = 0.0
    depot = 0
    stations: List[int] = []
    demands: List[int] = []
    name = path.name

    mode = None
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue

            if line.startswith("Name"):
                name = line.split(":", 1)[1].strip()
                continue
            if line.startswith("VEHICLES"):
                vehicles = int(line.split(":", 1)[1])
                continue
            if line.startswith("CAPACITY"):
                capacity = int(float(line.split(":", 1)[1]))
                continue
            if line.startswith("ENERGY_CAPACITY"):
                energy_capacity = float(line.split(":", 1)[1])
                continue

            if line == "NODE_COORD_SECTION":
                mode = "coords"
                continue
            if line == "DEMAND_SECTION":
                mode = "demand"
                continue
            if line == "STATIONS_COORD_SECTION":
                mode = "stations"
                continue
            if line == "DEPOT_SECTION":
                mode = "depot"
                continue
            if line == "EOF":
                break

            if mode == "coords":
                continue
            if mode == "demand":
                parts = line.split()
                if len(parts) >= 2:
                    index = int(parts[0])
                    demand = int(float(parts[1]))
                    while len(demands) <= index:
                        demands.append(0)
                    demands[index] = demand
                continue
            if mode == "stations":
                stations.append(int(line))
                continue
            if mode == "depot":
                if line == "-1":
                    mode = None
                else:
                    depot = int(line)
                continue

    return {
        "name": name,
        "vehicles": vehicles,
        "capacity": capacity,
        "energy_capacity": energy_capacity,
        "depot": depot,
        "stations": stations,
        "demands": demands or [0],
    }


def format_routes(solution: Solution) -> str:
    lines = []
    for route in solution.routes:
        joined = ",".join(str(node) for node in route)
        lines.append(joined)
    lines.append(f"{solution.cost:.6f}")
    return "\n".join(lines)


def read_ground_truth_cost(base_path: Path) -> Optional[float]:
    gt_path = base_path.with_name(base_path.name + "-evrp-time.txt")
    if not gt_path.exists():
        return None
    last_value: Optional[str] = None
    with gt_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                last_value = stripped
    if last_value is None:
        return None
    try:
        return float(last_value.split()[-1])
    except ValueError:
        return None


def extract_tensor_id(name: str) -> str:
    parts = name.split("-")
    for part in reversed(parts):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            return digits
    digits = "".join(ch for ch in name if ch.isdigit())
    return digits or name


def format_vehicle_routes(routes: Sequence[Sequence[int]]) -> List[str]:
    formatted: List[str] = []
    for idx, route in enumerate(routes, start=1):
        if len(route) <= 1:
            continue
        route_str = " -> ".join(str(node) for node in route)
        formatted.append(f" Vehicle {idx}: {route_str}")
    return formatted


def format_batch_entry(base_name: str, solution: Solution, gt_cost: Optional[float]) -> str:
    identifier = extract_tensor_id(base_name)
    lines: List[str] = [f"Instance tensor([[{identifier}]], device='cuda:0'):"]
    lines.append(f"Cost: {solution.cost:.4f}")
    if gt_cost is not None:
        lines.append(f"GT Cost: {gt_cost:.4f}")
        if gt_cost > 1e-9:
            gap = (solution.cost - gt_cost) / gt_cost * 100.0
            lines.append(f"Gap: {gap:.2f}%")
        else:
            lines.append("Gap: N/A")
    else:
        lines.append("GT Cost: N/A")
        lines.append("Gap: N/A")

    lines.append("Best path found:")
    vehicle_lines = format_vehicle_routes(solution.routes)
    if vehicle_lines:
        lines.extend(vehicle_lines)
    else:
        lines.append(" Vehicle 1: (no route constructed)")

    if solution.unserved:
        lines.append(f" Unserved customers: {','.join(map(str, solution.unserved))}")
    if solution.energy_violation > 1e-6:
        lines.append(f" Energy violation: {solution.energy_violation:.3f}")
    if solution.time_violation > 1e-6:
        lines.append(f" Time violation: {solution.time_violation:.3f}")
    if solution.capacity_violation > 1e-6:
        lines.append(f" Capacity violation: {solution.capacity_violation:.3f}")

    lines.append("-" * 50)
    return "\n".join(lines)


def format_error_entry(base_name: str, error: Exception) -> str:
    identifier = extract_tensor_id(base_name)
    return "\n".join(
        [
            f"Instance tensor([[{identifier}]], device='cuda:0'):",
            f"Error: {error}",
            "-" * 50,
        ]
    )


def solve_instance_path(
    base_path: Path,
    heuristic_path: Optional[Path],
    args: argparse.Namespace,
    *,
    progress_label: Optional[str] = None,
) -> Tuple[Solution, EVRPInstance, float]:
    instance = load_instance(base_path, heuristic_path)
    solver = ACOEvrpSolver(
        instance,
        seed=args.seed,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        q0=args.q0,
        num_ants=args.ants,
        time_relaxation=args.time_relaxation,
        penalty_update_interval=args.penalty_update_interval,
        # HGS options from CLI
        hgs_pool_size=args.hgs_pool_size,
        hgs_elite_size=args.hgs_elite_size,
        hgs_min_bpd=args.hgs_min_bpd,
        education_iter=args.education_iter,
        education_max_neighbors=args.education_max_neighbors,
        enable_swap_star=args.enable_swap_star,
        swap_star_iterations=args.swap_star_iterations,
        swap_star_sample_size=args.swap_star_sample_size,
        swap_star_removal_rate=args.swap_star_removal_rate,
        swap_star_min_remove=args.swap_star_min_remove,
        enable_split_repair=args.enable_split_repair,
        show_progress=not getattr(args, "no_progress", False),
        progress_label=progress_label or base_path.name,
    )
    start_ts = time.perf_counter()
    best = solver.run(iterations=args.iterations)
    runtime = time.perf_counter() - start_ts
    return best, instance, runtime


def process_folder(folder: Path, output_path: Path, heuristic_path: Optional[Path], args: argparse.Namespace) -> None:
    energy_files = sorted(folder.rglob("*-energy.txt"))
    if not energy_files:
        raise FileNotFoundError(f"No energy matrices found in folder {folder}")

    completed_headers: set[str] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped.startswith("Instance tensor("):
                    completed_headers.add(stripped)

    error_path = output_path.parent / "error.txt"

    written = 0
    skipped = 0

    def append_entry(text: str) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        needs_sep = output_path.exists() and output_path.stat().st_size > 0
        with output_path.open("a", encoding="utf-8") as handle:
            if needs_sep:
                handle.write("\n\n")
            handle.write(text)
            handle.write("\n")

    def append_error(identifier: str) -> None:
        error_path.parent.mkdir(parents=True, exist_ok=True)
        with error_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{identifier}\n")

    for energy_file in energy_files:
        base_name = energy_file.stem.replace("-energy", "")
        base_dir = energy_file.parent
        base_path = base_dir / base_name
        data_file = base_path.with_suffix(".txt")
        if not data_file.exists():
            continue
        try:
            rel_label = base_path.relative_to(folder)
        except ValueError:
            rel_label = base_path.name
        identifier = extract_tensor_id(str(rel_label))
        header_line = f"Instance tensor([[{identifier}]], device='cuda:0'):"
        if header_line in completed_headers:
            skipped += 1
            continue

        attempts = 3
        last_exception: Optional[Exception] = None
        final_solution: Optional[Solution] = None
        final_gt: Optional[float] = None
        success = False

        for attempt in range(attempts):
            try:
                solution, _, runtime = solve_instance_path(
                    base_path, heuristic_path, args, progress_label=str(rel_label)
                )
            except Exception as exc:
                last_exception = exc
                continue

            final_solution = solution
            final_gt = read_ground_truth_cost(base_path)

            acceptable = solution.feasible and solution.time_violation <= 1e-6
            if acceptable:
                success = True
                break

        if not success:
            append_error(identifier)
            if last_exception is not None:
                entry = format_error_entry(str(rel_label), last_exception)
                append_entry(entry)
                written += 1
            continue

        assert final_solution is not None
        entry = format_batch_entry(str(rel_label), final_solution, final_gt)
        append_entry(entry)
        completed_headers.add(header_line)
        written += 1

    print(
        f"Processed {written} new entries (skipped {skipped}). Results saved to {output_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACO + HGS-style EVRP solver.")
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--instance",
        help="Path prefix to the EVRP instance (for X.txt, supply the path up to X).",
    )
    target_group.add_argument(
        "--folder",
        help="Directory containing multiple EVRP instances to solve in batch mode.",
    )
    parser.add_argument(
        "--heuristic",
        help="Optional path to a heuristic matrix; defaults to the reciprocal of the energy matrix.",
    )
    parser.add_argument("--iterations", type=int, default=200, help="Number of ACO iterations to perform.")
    parser.add_argument("--ants", type=int, help="Number of ants per iteration.")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Pheromone influence exponent.")
    parser.add_argument("--beta", type=float, default=2.0, help="Heuristic influence exponent.")
    parser.add_argument("--rho", type=float, default=0.1, help="Pheromone evaporation rate.")
    parser.add_argument("--q0", type=float, default=0.2, help="Greedy selection probability.")
    parser.add_argument("--no-progress", action="store_true", help="Disable iteration progress bar output.")
    parser.add_argument(
        "--time-relaxation",
        type=float,
        default=0,
        help="Relative relaxation factor applied to the 8h time limit during solution construction (e.g. 0.15 allows +15%).",
    )
    parser.add_argument(
        "--penalty-update-interval",
        type=int,
        default=25,
        help="Iterations between adaptive penalty updates inspired by HGS.",
    )
    # HGS-style params
    parser.add_argument("--hgs-pool-size", type=int, default=30, help="Population size for HGS-style pool.")
    parser.add_argument("--hgs-elite-size", type=int, default=6, help="Elite size (used for pheromone reinforcement).")
    parser.add_argument("--hgs-min-bpd", type=float, default=0.15, help="Minimum broken-pairs distance to accept new solutions.")
    parser.add_argument("--education-iter", type=int, default=200, help="Max iterations of education (local search).")
    parser.add_argument("--education-max-neighbors", type=int, default=200, help="Neighborhood cap per education step.")
    parser.add_argument("--enable-swap-star", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable SWAP* multi-level improvement after construction.")
    parser.add_argument("--swap-star-iterations", type=int, default=4,
                        help="Number of passes through the SWAP* neighborhoods.")
    parser.add_argument("--swap-star-sample-size", type=int, default=60,
                        help="Maximum sampled neighbors per SWAP* operator.")
    parser.add_argument("--swap-star-removal-rate", type=float, default=0.2,
                        help="Fraction of customers removed during SWAP* LNS (0-0.8).")
    parser.add_argument("--swap-star-min-remove", type=int, default=2,
                        help="Minimum customers removed during SWAP* LNS.")
    parser.add_argument("--enable-split-repair", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable Split-like greedy route reconstruction before local search.")
    parser.add_argument(
        "--output",
        help="Output report path when using --folder (defaults to aco_batch_results.txt inside the folder).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heuristic_path = Path(args.heuristic) if args.heuristic else None

    if args.folder:
        folder = Path(args.folder)
        output_path = Path(args.output) if args.output else folder / "aco_batch_results.txt"
        process_folder(folder, output_path, heuristic_path, args)
        return

    base_path = Path(args.instance)
    best, _, runtime = solve_instance_path(base_path, heuristic_path, args, progress_label=base_path.name)
    print(format_routes(best))
    if not best.feasible:
        print("# Warning: best solution violates constraints.", flush=True)
        if best.unserved:
            print("# Unserved customers:", ",".join(map(str, best.unserved)))
    if best.energy_violation > 1e-6:
        print(f"# Energy violation (total deficit): {best.energy_violation:.3f}")
    if best.time_violation > 1e-6:
        print(f"# Time violation (minutes): {best.time_violation:.3f}")
    if best.capacity_violation > 1e-6:
        print(f"# Capacity violation (units): {best.capacity_violation:.3f}")

    gt_cost = read_ground_truth_cost(base_path)
    if gt_cost is not None:
        print(f"# GT Cost: {gt_cost:.6f}")
        if gt_cost > 1e-9:
            gap = (best.cost - gt_cost) / gt_cost * 100.0
            print(f"# Gap: {gap:.2f}%")
        else:
            print("# Gap: undefined (GT cost <= 0)")
    print(f"# Runtime: {runtime:.2f}s")


if __name__ == "__main__":
    main()
