package org.sbpo2025.challenge;

import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.time.StopWatch;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 120; // 2 min
    protected List<Map<Integer, Integer>> orders;
    protected List<Map<Integer, Integer>> aisles;
    protected int nItems;
    protected int waveSizeLB;
    protected int waveSizeUB;
    protected Map<Neighborhood, Double> weightedNeighborhoods = new HashMap<>(
            Map.of(
                    new RemoveAisle(), 1.0,
                    new AddOrder(), 1.0,
                    new Shake(), 1.0,
                    new OneAisleMultipleOrders(), 1.0,
                    new OneOrderMultipleAisles(), 1.0,
                    new SwapOrder(), 1.0,
                    new SwapAisle(), 1.0));
    protected Map<Neighborhood, Double> weightedChunkNeighborhoods = new HashMap<>(
            Map.of(
                    new ChunkShake(), 15.0));


    public double alpha;
    public int shakeCooldown;
    public int greedyCooldown;
    public int k;

    public ChallengeSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB,
            int waveSizeUB, double alpha, int shakeCooldown, int greedyCooldown, int k) {
        this.orders = orders;
        this.aisles = aisles;
        this.nItems = nItems;
        this.waveSizeLB = waveSizeLB;
        this.waveSizeUB = waveSizeUB;


        this.alpha = alpha;
        this.shakeCooldown = shakeCooldown;
        this.greedyCooldown = greedyCooldown;
        this.k = k;

        if (orders.size() + aisles.size() > 5000) {
            this.weightedNeighborhoods.putAll(this.weightedChunkNeighborhoods);
        }

        System.setProperty("java.util.concurrent.ForkJoinPool.common.parallelism", "6");

    }

    public static class RVNDSolution {
        public List<Integer> orders;
        public List<Integer> aisles;

        public RVNDSolution(List<Integer> orders, List<Integer> aisles) {
            this.orders = orders;
            this.aisles = aisles;
        }

        public List<Integer> orders() {
            return orders;
        }

        public List<Integer> aisles() {
            return aisles;
        }

        @Override
        public String toString() {
            return "RVNDSolution{" +
                    "orders=" + orders +
                    ", aisles=" + aisles +
                    '}';
        }
    }

    public ChallengeSolution solve(StopWatch stopWatch) {

        RVNDSolution bestSolution = constructGreedyRandomizedSolution(alpha);
        double bestQuality = computeObjectiveFunction(bestSolution);

        int currentIteration = 0;
        int maxIterations = Integer.MAX_VALUE;
        Shake shake = new Shake();

        RVNDSolution currentSolution = bestSolution;
        while (MAX_RUNTIME - stopWatch.getTime(TimeUnit.SECONDS) > 0 && currentIteration < maxIterations) {
            currentIteration++;
            System.out.println("Iteration: " + currentIteration);
            if (currentIteration % greedyCooldown == 0) {
                currentSolution = constructGreedyRandomizedSolution(alpha);
            } else if (currentIteration % shakeCooldown == 0) {
                currentSolution = shake.explore(bestSolution, bestQuality, true);
            }

            double currentQuality = bestQuality;

            currentSolution = randomVariableNeighborhoodDescent(currentSolution, currentQuality, k);
            currentQuality = computeObjectiveFunction(currentSolution);

            if (currentQuality > bestQuality) {
                bestQuality = currentQuality;
                bestSolution = currentSolution;
            }
        }

        return new ChallengeSolution(new HashSet<>(bestSolution.orders()), new HashSet<>(bestSolution.aisles()));
    }

    private RVNDSolution constructGreedyRandomizedSolution(double alpha) {
        Random rand = new Random();
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();
        Map<Integer, Integer> availableItems = new HashMap<>();

        for (int i = 0; i < aisles.size(); i++) {
            selectedAisles.add(i);
            for (Map.Entry<Integer, Integer> entry : aisles.get(i).entrySet()) {
                availableItems.put(entry.getKey(), entry.getValue() + availableItems.getOrDefault(entry.getKey(), 0));
            }
        }

        List<Integer> candidateOrders = new ArrayList<>();
        Map<Integer, Integer> orderItemCount = new HashMap<>();
        for (int i = 0; i < orders.size(); i++) {
            int totalItems = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
            orderItemCount.put(i, totalItems);
            candidateOrders.add(i);
        }

        int totalSelectedItems = 0;
        List<Integer> RCL = buildRCL(candidateOrders, orderItemCount, alpha);

        while (!candidateOrders.isEmpty()) {
            if (RCL.isEmpty()) {
                RCL = buildRCL(candidateOrders, orderItemCount, alpha);
            }
            if (RCL.isEmpty())
                break;
            int selectedOrder = RCL.remove(rand.nextInt(RCL.size()));
            int selectedOrderItems = orderItemCount.get(selectedOrder);
            if (totalSelectedItems + selectedOrderItems > waveSizeUB) {
                candidateOrders.remove(Integer.valueOf(selectedOrder));
                continue;
            }
            Map<Integer, Integer> order = orders.get(selectedOrder);
            boolean canAdd = true;
            for (Map.Entry<Integer, Integer> entry : order.entrySet()) {
                int item = entry.getKey();
                int needed = entry.getValue();
                if (availableItems.getOrDefault(item, 0) < needed) {
                    canAdd = false;
                    break;
                }
            }
            if (canAdd) {
                for (Map.Entry<Integer, Integer> entry : order.entrySet()) {
                    int item = entry.getKey();
                    int needed = entry.getValue();
                    availableItems.put(item, availableItems.get(item) - needed);
                }
                selectedOrders.add(selectedOrder);
                totalSelectedItems += selectedOrderItems;
            }
            candidateOrders.remove(Integer.valueOf(selectedOrder));
        }

        return new RVNDSolution(new ArrayList<>(selectedOrders), new ArrayList<>(selectedAisles));
    }

    private HashMap<Integer, Integer> getItemsLeftInAisles(Set<Integer> selectedOrders, Set<Integer> selectedAisles) {
        ConcurrentMap<Integer, Integer> itemsLeftInAisles = new ConcurrentHashMap<>();

        // Adds all aisles items to the itemsLeftInAisles
        selectedAisles.parallelStream().forEach(aisle -> {
            aisles.get(aisle).forEach((item, quantity) -> {
                itemsLeftInAisles.merge(item, quantity, Integer::sum);
            });
        });

        // Subtracts the items that were selected in the orders
        selectedOrders.parallelStream().forEach(order -> {
            orders.get(order).forEach((item, quantity) -> {
                itemsLeftInAisles.merge(item, -quantity, Integer::sum);
            });
        });

        return new HashMap<>(itemsLeftInAisles);
    }

    private List<Integer> buildRCL(List<Integer> candidateOrders, Map<Integer, Integer> orderItemCount, double alpha) {
        int maxItems = candidateOrders.parallelStream().mapToInt(orderItemCount::get).max().orElse(0);
        int minItems = candidateOrders.parallelStream().mapToInt(orderItemCount::get).min().orElse(0);
        double threshold = maxItems - alpha * (maxItems - minItems);

        List<Integer> RCL = candidateOrders.parallelStream()
                .filter(order -> orderItemCount.get(order) >= threshold)
                .collect(Collectors.toList());

        return RCL;
    }

    class Neighborhood {
        protected final int MAX_ITERATIONS = 50;

        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality, boolean onlyFeasible) {
            return null;
        }

        boolean move(RVNDSolution solution, int iterator) {
            return false;
        }
    }

    class RemoveAisle extends Neighborhood {

        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality, boolean onlyFeasible) {
            RVNDSolution bestSolution = initialSolution;

            bestSolution = IntStream.range(0, initialSolution.aisles().size()).limit(this.MAX_ITERATIONS).parallel()
                    .mapToObj((aisleIndex) -> {
                        RVNDSolution currentSolution = new RVNDSolution(
                                new ArrayList<>(initialSolution.orders()),
                                new ArrayList<>(initialSolution.aisles()));

                        if (!this.move(currentSolution, aisleIndex) && onlyFeasible) {
                            return null;
                        }

                        double currentQuality = computeObjectiveFunction(currentSolution);

                        return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
                    }).filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey)
                    .orElse(bestSolution);

            return bestSolution;
        }

        // A move configures a removal of an aisle and, from there, the removal of any
        // other aisle that can be removed
        @Override
        boolean move(RVNDSolution solution, int iterator) {
            if (iterator >= solution.aisles().size())
                return false;

            solution.aisles().remove(iterator);

            if (!isSolutionFeasible(solution, false))
                return false;

            int nextRemovableAisle = getNextRemovableAisle(solution);
            while (nextRemovableAisle != -1) {
                solution.aisles().remove(nextRemovableAisle);
                nextRemovableAisle = getNextRemovableAisle(solution);
            }

            return true;
        }

        int getNextRemovableAisle(RVNDSolution solution) {
            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(new HashSet<>(solution.orders()),
                    new HashSet<>(solution.aisles()));

            for (int i = 0; i < solution.aisles().size(); i++) {

                int aisleIndex = solution.aisles().get(i);
                HashMap<Integer, Integer> aisleItems = new HashMap<>(aisles.get(aisleIndex));
                boolean canRemove = true;

                for (Map.Entry<Integer, Integer> entry : aisleItems.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    if (itemsLeftInAisles.getOrDefault(item, 0) < quantity) {
                        canRemove = false;
                        break;
                    }
                }

                if (canRemove) {
                    return i;
                }
            }
            return -1;
        }

    }

    class AddOrder extends Neighborhood {
        ArrayList<Integer> candidateOrders;

        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality, boolean onlyFeasible) {
            RVNDSolution bestSolution = initialSolution;

            this.candidateOrders = IntStream.range(0, orders.size())
                    .boxed()
                    .parallel()
                    .filter(order -> !initialSolution.orders().contains(order))
                    .collect(Collectors.toCollection(ArrayList::new));

            Collections.shuffle(candidateOrders);

            bestSolution = IntStream.range(0, candidateOrders.size())
                    .limit(this.MAX_ITERATIONS)
                    .parallel()
                    .mapToObj(orderIndex -> {
                        RVNDSolution currentSolution = new RVNDSolution(
                                new ArrayList<>(initialSolution.orders()),
                                new ArrayList<>(initialSolution.aisles()));

                        if (!this.move(currentSolution, orderIndex) && onlyFeasible)
                            return null;

                        double currentQuality = computeObjectiveFunction(currentSolution);
                        return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
                    })
                    .filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey)
                    .orElse(bestSolution);

            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution, int iterator) {

            if (iterator >= candidateOrders.size())
                return false;

            solution.orders().add(candidateOrders.get(iterator));

            if (!isSolutionFeasible(solution, false))
                return false;

            int nextCandidateOrder = getNextCandidateOrderToAdd(solution);
            while (nextCandidateOrder != -1) {

                solution.orders().add(candidateOrders.get(nextCandidateOrder));
                nextCandidateOrder = getNextCandidateOrderToAdd(solution);
            }

            return true;
        }

        int getNextCandidateOrderToAdd(RVNDSolution solution) {
            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(new HashSet<>(solution.orders()),
                    new HashSet<>(solution.aisles()));

            int totalItems = solution.orders().stream()
                    .mapToInt(order -> orders.get(order).values().stream().mapToInt(Integer::intValue).sum()).sum();

            for (int i = 0; i < candidateOrders.size(); i++) {
                int orderIndex = candidateOrders.get(i);
                if (solution.orders().contains(orderIndex))
                    continue;
                HashMap<Integer, Integer> orderItems = new HashMap<>(orders.get(orderIndex));

                if (orderItems.values().stream().mapToInt(Integer::intValue).sum() + totalItems > waveSizeUB)
                    continue;

                boolean canAdd = true;

                for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    if (itemsLeftInAisles.getOrDefault(item, 0) < quantity) {
                        canAdd = false;
                        break;
                    }
                }

                if (canAdd)
                    return i;
            }

            return -1;
        }
    }

    class Shake extends Neighborhood {
        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality, boolean onlyFeasible) {

            RVNDSolution bestSolution = initialSolution;

            bestSolution = IntStream.range(0, this.MAX_ITERATIONS).parallel().mapToObj((orderIndex) -> {
                RVNDSolution currentSolution = new RVNDSolution(
                        new ArrayList<>(initialSolution.orders()),
                        new ArrayList<>(initialSolution.aisles()));

                if (!this.move(currentSolution, orderIndex) && onlyFeasible) {
                    return null;
                }

                double currentQuality = computeObjectiveFunction(currentSolution);

                return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
            }).filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey) // Extrai a solução
                    .orElse(bestSolution);

            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution, int iterator) {

            Random rand = new Random();

            int ordersSize = solution.orders().size();
            int aislesSize = solution.aisles().size();

            int numberOfOrdersToRemove = ordersSize > 0 ? rand.nextInt((int) Math.max(3, ordersSize * 0.1)) : 0;
            int numberOfAislesToRemove = aislesSize > 0 ? rand.nextInt((int) Math.max(3, aislesSize * 0.1)) : 0;

            for (int i = 0; i < numberOfOrdersToRemove && !solution.orders().isEmpty(); i++) {
                int orderIndex = rand.nextInt(solution.orders().size());
                solution.orders().remove(orderIndex);
            }

            for (int i = 0; i < numberOfAislesToRemove && !solution.aisles().isEmpty(); i++) {
                int aisleIndex = rand.nextInt(solution.aisles.size());
                solution.aisles().remove(aisleIndex);
            }

            ordersSize = solution.orders().size();
            aislesSize = solution.aisles().size();

            int numberOfOrdersToAdd = orders.size() - ordersSize > 0
                    ? rand.nextInt((int) Math.max(3, (orders.size() - ordersSize) * 0.1))
                    : 0;
            int numberOfAislesToAdd = aisles.size() - aislesSize > 0
                    ? rand.nextInt((int) Math.max(3, (aisles.size() - aislesSize) * 0.1))
                    : 0;

            for (int i = 0; i < numberOfOrdersToAdd; i++) {
                int orderIndex = rand.nextInt(orders.size());
                if (solution.orders().contains(orderIndex))
                    continue;

                solution.orders().add(orderIndex);
            }

            for (int i = 0; i < numberOfAislesToAdd; i++) {
                int aisleIndex = rand.nextInt(aisles.size());
                if (solution.aisles().contains(aisleIndex))
                    continue;

                solution.aisles().add(aisleIndex);
            }

            return isSolutionFeasible(solution, false);
        }

    }

    class ChunkShake extends Neighborhood {
        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality, boolean onlyFeasible) {
            RVNDSolution bestSolution = initialSolution;

            bestSolution = IntStream.range(0, 10).parallel().mapToObj((orderIndex) -> {
                RVNDSolution currentSolution = new RVNDSolution(
                        new ArrayList<>(initialSolution.orders()),
                        new ArrayList<>(initialSolution.aisles()));

                if (!this.move(currentSolution, orderIndex) && onlyFeasible) {
                    return null;
                }

                double currentQuality = computeObjectiveFunction(currentSolution);

                return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
            }).filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey) // Extrai a solução
                    .orElse(bestSolution);

            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution, int iterator) {

            Random rand = new Random();

            int ordersSize = orders.size();
            int aislesSize = aisles.size();

            long numberOfOrdersToRemove = Math.round(ordersSize * 0.1);
            long numberOfAislesToRemove = Math.round(aislesSize * 0.1);

            for (int i = 0; i < numberOfOrdersToRemove && !solution.orders().isEmpty(); i++) {
                int orderIndex = rand.nextInt(solution.orders().size());

                solution.orders().remove(orderIndex);
            }

            for (int i = 0; i < numberOfAislesToRemove && !solution.aisles().isEmpty(); i++) {
                int aisleIndex = rand.nextInt(solution.aisles.size());

                solution.aisles().remove(aisleIndex);
            }

            ordersSize = solution.orders().size();
            aislesSize = solution.aisles().size();

            Long numberOfOrdersToAdd = orders.size() - ordersSize > 0
                    ? Math.round(ordersSize * 0.1)
                    : 0;
            Long numberOfAislesToAdd = aisles.size() - aislesSize > 0
                    ? Math.round(aislesSize * 0.1)
                    : 0;

            for (int i = 0; i < numberOfOrdersToAdd; i++) {
                int orderIndex = rand.nextInt(orders.size());
                if (solution.orders().contains(orderIndex))
                    continue;

                solution.orders().add(orderIndex);
            }

            for (int i = 0; i < numberOfAislesToAdd; i++) {
                int aisleIndex = rand.nextInt(aisles.size());
                if (solution.aisles().contains(aisleIndex))
                    continue;

                solution.aisles().add(aisleIndex);
            }

            return isSolutionFeasible(solution, false);
        }

    }

    class OneAisleMultipleOrders extends Neighborhood {
        List<Integer> remainingOrders;
        List<Integer> remainingAisles;

        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality, boolean onlyFeasible) {

            RVNDSolution bestSolution = initialSolution;

            this.remainingAisles = IntStream.range(0, aisles.size())
                    .boxed()
                    .parallel()
                    .filter(aisle -> !initialSolution.aisles().contains(aisle))
                    .sorted(Comparator.comparingInt(
                            (Integer aisle) -> aisles.get(aisle).values().stream().mapToInt(Integer::intValue).sum())
                            .reversed())
                    .collect(Collectors.toList());

            this.remainingOrders = IntStream.range(0, orders.size())
                    .boxed()
                    .parallel()
                    .filter(order -> !initialSolution.orders().contains(order))
                    .sorted(Comparator.comparingInt(
                            (Integer order) -> orders.get(order).values().stream().mapToInt(Integer::intValue).sum())
                            .reversed())
                    .collect(Collectors.toList());

            Collections.shuffle(remainingOrders);

            bestSolution = IntStream.range(0, remainingAisles.size()).parallel().mapToObj((aisleIndex) -> {
                RVNDSolution currentSolution = new RVNDSolution(
                        new ArrayList<>(initialSolution.orders()),
                        new ArrayList<>(initialSolution.aisles()));

                if (!this.move(currentSolution, aisleIndex) && onlyFeasible) {
                    return null;
                }

                double currentQuality = computeObjectiveFunction(currentSolution);

                return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
            }).filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey) // Extrai a solução
                    .orElse(bestSolution);

            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution, int iterator) {

            if (iterator >= this.remainingAisles.size())
                return false;

            solution.aisles().add(remainingAisles.get(iterator));

            int nextAddableOrder = getNextCandidateOrderToAdd(solution);

            while (nextAddableOrder != -1) {
                solution.orders().add(remainingOrders.get(nextAddableOrder));

                nextAddableOrder = getNextCandidateOrderToAdd(solution);
            }

            return isSolutionFeasible(solution, false);
        }

        int getNextCandidateOrderToAdd(RVNDSolution solution) {
            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(new HashSet<>(solution.orders()),
                    new HashSet<>(solution.aisles()));

            int totalItems = solution.orders().stream()
                    .mapToInt(order -> orders.get(order).values().stream().mapToInt(Integer::intValue).sum()).sum();

            for (int i = 0; i < remainingOrders.size(); i++) {
                int orderIndex = remainingOrders.get(i);
                if (solution.orders().contains(orderIndex))
                    continue;
                HashMap<Integer, Integer> orderItems = new HashMap<>(orders.get(orderIndex));

                if (orderItems.values().stream().mapToInt(Integer::intValue).sum() + totalItems > waveSizeUB)
                    continue;

                boolean canAdd = true;

                for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    if (itemsLeftInAisles.getOrDefault(item, 0) < quantity) {
                        canAdd = false;
                        break;
                    }
                }

                if (canAdd)
                    return i;
            }

            return -1;
        }
    }

    class OneOrderMultipleAisles extends Neighborhood {
        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality, boolean onlyFeasible) {
            
            RVNDSolution bestSolution = initialSolution;

            initialSolution.orders().sort(Comparator
                    .comparingInt(order -> orders.get(order).values().stream().mapToInt(Integer::intValue).sum()));

            bestSolution = IntStream.range(0, initialSolution.orders().size()).limit(this.MAX_ITERATIONS).parallel()
                    .mapToObj((orderIndex) -> {
                        RVNDSolution currentSolution = new RVNDSolution(
                                new ArrayList<>(initialSolution.orders()),
                                new ArrayList<>(initialSolution.aisles()));

                        if (!this.move(currentSolution, orderIndex) && onlyFeasible) {
                            return null;
                        }

                        double currentQuality = computeObjectiveFunction(currentSolution);

                        return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
                    }).filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey)
                    .orElse(bestSolution);

            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution, int iterator) {
            if (iterator >= solution.orders().size())
                return false;

            solution.orders().remove(iterator);

            int nextRemovableAisle = getNextRemovableAisle(solution);

            while (nextRemovableAisle != -1) {
                solution.aisles().remove(nextRemovableAisle);
                nextRemovableAisle = getNextRemovableAisle(solution);
            }

            return isSolutionFeasible(solution, false);
        }

        int getNextRemovableAisle(RVNDSolution solution) {
            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(new HashSet<>(solution.orders()),
                    new HashSet<>(solution.aisles()));

            for (int i = 0; i < solution.aisles().size(); i++) {

                int aisleIndex = solution.aisles().get(i);
                HashMap<Integer, Integer> aisleItems = new HashMap<>(aisles.get(aisleIndex));
                boolean canRemove = true;

                for (Map.Entry<Integer, Integer> entry : aisleItems.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    if (itemsLeftInAisles.getOrDefault(item, 0) < quantity) {
                        canRemove = false;
                        break;
                    }
                }

                if (canRemove)
                    return i;
            }
            return -1;
        }
    }

    class SwapOrder extends Neighborhood {
        List<Integer> remainingOrders;

        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality, boolean onlyFeasible) {
            RVNDSolution bestSolution = initialSolution;

            this.remainingOrders = IntStream.range(0, orders.size())
                    .boxed()
                    .parallel()
                    .filter(order -> !initialSolution.orders().contains(order))
                    .collect(Collectors.toList());

            Collections.shuffle(remainingOrders);

            this.remainingOrders = this.remainingOrders.stream()
                .limit(100)
                .collect(Collectors.toList());

            bestSolution = IntStream.range(0, Math.min(this.MAX_ITERATIONS, initialSolution.orders().size()))
                    .parallel()
                    .mapToObj((orderIndex) -> {
                        RVNDSolution currentSolution = new RVNDSolution(
                                new ArrayList<>(initialSolution.orders()),
                                new ArrayList<>(initialSolution.aisles()));

                        if (!this.move(currentSolution, orderIndex) && onlyFeasible) {
                            return null;
                        }

                        double currentQuality = computeObjectiveFunction(currentSolution);

                        return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
                    }).filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey)
                    .orElse(bestSolution);

            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution, int iterator) {
            if (iterator >= solution.orders().size() || remainingOrders.isEmpty())
                return false;
            
            double bestQuality = -1;
            int originalOrder = solution.orders().get(iterator);
            for (int i = 0; i < remainingOrders.size(); i++) {
                int orderToAdd = remainingOrders.get(i);
                solution.orders().set(iterator, orderToAdd);

                if (isSolutionFeasible(solution, false)) {
                    double currentQuality = computeObjectiveFunction(solution);
                    if (currentQuality > bestQuality) {
                        bestQuality = currentQuality;
                        originalOrder = orderToAdd;
                    } else {
                        solution.orders().set(iterator, originalOrder);
                    }
                } else {
                    solution.orders().set(iterator, originalOrder);
                }
            }

            return bestQuality != -1;
        }

    }

    class SwapAisle extends Neighborhood {
        List<Integer> remainingAisles;

        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality, boolean onlyFeasible) {
            RVNDSolution bestSolution = initialSolution;

            this.remainingAisles = IntStream.range(0, aisles.size())
                    .boxed()
                    .parallel()
                    .filter(aisle -> !initialSolution.aisles().contains(aisle))
                    .collect(Collectors.toList());

            Collections.shuffle(remainingAisles);

            this.remainingAisles = this.remainingAisles.stream()
                .limit(100)
                .collect(Collectors.toList());

            bestSolution = IntStream.range(0, Math.min(this.MAX_ITERATIONS, initialSolution.aisles().size()))
                    .parallel()
                    .mapToObj((aisleIndex) -> {
                        RVNDSolution currentSolution = new RVNDSolution(
                                new ArrayList<>(initialSolution.orders()),
                                new ArrayList<>(initialSolution.aisles()));

                        if (!this.move(currentSolution, aisleIndex) && onlyFeasible) {
                            return null;
                        }

                        double currentQuality = computeObjectiveFunction(currentSolution);

                        return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
                    }).filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey)
                    .orElse(bestSolution);

            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution, int iterator) {
            if (iterator >= solution.aisles().size() || remainingAisles.isEmpty())
                return false;
            
            double bestQuality = -1;
            int originalAisle = solution.aisles().get(iterator);
            for (int i = 0; i < remainingAisles.size(); i++) {
                int aisleToAdd = remainingAisles.get(i);
                solution.aisles().set(iterator, aisleToAdd);

                if (isSolutionFeasible(solution, false)) {
                    double currentQuality = computeObjectiveFunction(solution);
                    if (currentQuality > bestQuality) {
                        bestQuality = currentQuality;
                        originalAisle = aisleToAdd;
                    } else {
                        solution.aisles().set(iterator, originalAisle);
                    }
                } else {
                    solution.aisles().set(iterator, originalAisle);
                }
            }

            return bestQuality != -1;
        }

    }

    private RVNDSolution randomVariableNeighborhoodDescent(RVNDSolution solution, double quality, int k) {
        RVNDSolution bestSolution = solution;
        double bestQuality = quality;

        Collections.shuffle(solution.orders());
        Collections.shuffle(solution.aisles());

        RVNDSolution currentSolution = bestSolution;

        List<Neighborhood> neighborhoods = selectWeightNeighborhoods(k);
        for (Neighborhood neighborhood : neighborhoods) {
            System.out.println("Exploring neighborhood: " + neighborhood.getClass().getSimpleName());
            currentSolution = neighborhood.explore(currentSolution, bestQuality, false);
            double currentQuality = computeObjectiveFunction(currentSolution);

            if (currentQuality > bestQuality && isSolutionFeasible(currentSolution, false)) {
                bestQuality = currentQuality;
                bestSolution = currentSolution;
                weightedNeighborhoods.put(neighborhood, Math.min(weightedNeighborhoods.get(neighborhood) * 1.2, 15.0));
            }

            for (Map.Entry<Neighborhood, Double> entry : weightedNeighborhoods.entrySet()) {
                double newWeight = Math.max(1, entry.getValue() * 0.99);

                weightedNeighborhoods.put(entry.getKey(), newWeight);
            }

        }  

        return bestSolution;
    }

    private List<Neighborhood> selectWeightNeighborhoods(int k) {
        List<Neighborhood> selected = new ArrayList<>();
        Map<Neighborhood, Double> copy = new HashMap<>(weightedNeighborhoods);
        Random rand = new Random();

        while (selected.size() < k && !copy.isEmpty()) {
            double totalWeight = copy.values().stream().mapToDouble(Double::doubleValue).sum();
            double r = rand.nextDouble() * totalWeight;
            double cumulative = 0.0;

            for (Map.Entry<Neighborhood, Double> entry : copy.entrySet()) {
                cumulative += entry.getValue();
                if (r <= cumulative) {
                    selected.add(entry.getKey());
                    copy.remove(entry.getKey()); // evita repetição
                    break;
                }
            }
        }

        return selected;
    }

    protected ChallengeSolution decodeRVNDSolution(RVNDSolution solution) {
        return new ChallengeSolution(new HashSet<>(solution.orders()), new HashSet<>(solution.aisles));
    }

    /*
     * Get the remaining time in seconds
     */
    protected long getRemainingTime(StopWatch stopWatch) {
        return Math.max(
                TimeUnit.SECONDS.convert(MAX_RUNTIME - stopWatch.getTime(TimeUnit.MILLISECONDS), TimeUnit.MILLISECONDS),
                0);
    }

    protected boolean isSolutionFeasible(RVNDSolution challengeSolution, boolean print) {
        List<Integer> selectedOrders = challengeSolution.orders();
        List<Integer> visitedAisles = challengeSolution.aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return false;
        }

        int[] totalUnitsPicked = new int[nItems];
        int[] totalUnitsAvailable = new int[nItems];

        // Calculate total units picked
        for (int order : selectedOrders) {
            for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                totalUnitsPicked[entry.getKey()] += entry.getValue();
            }
        }

        // Calculate total units available
        for (int aisle : visitedAisles) {
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                totalUnitsAvailable[entry.getKey()] += entry.getValue();
            }
        }

        // Check if the total units picked are within bounds
        int totalUnits = Arrays.stream(totalUnitsPicked).sum();
        if (totalUnits < waveSizeLB) {
            if (print) {
                System.out.println("Motive: totalUnits < waveSizeLB");
            }
            return false;
        }
        if (totalUnits > waveSizeUB) {
            if (print) {
                System.out.println("Motive: totalUnits > waveSizeUB");
            }
            return false;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {
                if (print) {
                    System.out.println("Motive: More Picked Items than offered by Aisles");
                }
                return false;
            }
        }

        return true;
    }

    protected int calculatePenalty(RVNDSolution challengeSolution, boolean print) {
        List<Integer> selectedOrders = challengeSolution.orders();
        List<Integer> visitedAisles = challengeSolution.aisles();

        int penalty = 0;

        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return 0;
        }

        int[] totalUnitsPicked = new int[nItems];
        int[] totalUnitsAvailable = new int[nItems];

        // Calculate total units picked
        for (int order : selectedOrders) {
            for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                totalUnitsPicked[entry.getKey()] += entry.getValue();
            }
        }

        // Calculate total units available
        for (int aisle : visitedAisles) {
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                totalUnitsAvailable[entry.getKey()] += entry.getValue();
            }
        }

        // Check if the total units picked are within bounds
        int totalUnits = Arrays.stream(totalUnitsPicked).sum();
        if (totalUnits < waveSizeLB) {
            penalty += (waveSizeLB - totalUnits) * 100;
        }
        if (totalUnits > waveSizeUB) {
            penalty += (totalUnits - waveSizeUB) * 100;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {
                penalty += (totalUnitsPicked[i] - totalUnitsAvailable[i]) * 100;
            }
        }

        return penalty;
    }

    protected double computeObjectiveFunction(RVNDSolution challengeSolution) {
        List<Integer> selectedOrders = challengeSolution.orders();
        List<Integer> visitedAisles = challengeSolution.aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return 0.0;
        }
        int totalUnitsPicked = 0;

        // Calculate total units picked
        for (int order : selectedOrders) {
            totalUnitsPicked += orders.get(order).values().stream()
                    .mapToInt(Integer::intValue)
                    .sum();
        }

        // Calculate the number of visited aisles
        int numVisitedAisles = visitedAisles.size();

        // Objective function: total units picked / number of visited aisles
        return ((double) totalUnitsPicked / numVisitedAisles) - calculatePenalty(challengeSolution, false);
    }

    protected double computeObjectiveFunction(ChallengeSolution challengeSolution) {
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return 0.0;
        }
        int totalUnitsPicked = 0;

        // Calculate total units picked
        for (int order : selectedOrders) {
            totalUnitsPicked += orders.get(order).values().stream()
                    .mapToInt(Integer::intValue)
                    .sum();
        }

        // Calculate the number of visited aisles
        int numVisitedAisles = visitedAisles.size();

        // Objective function: total units picked / number of visited aisles
        return ((double) totalUnitsPicked / numVisitedAisles);
    }

}
