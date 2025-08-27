package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 60000; // milliseconds; 5 s
    private final int NUM_THREADS = Runtime.getRuntime().availableProcessors();
    protected List<Map<Integer, Integer>> orders;
    protected List<Map<Integer, Integer>> aisles;
    protected int nItems;
    protected int waveSizeLB;
    protected int waveSizeUB;

    public ChallengeSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB, int waveSizeUB) {
        this.orders = orders;
        this.aisles = aisles;
        this.nItems = nItems;
        this.waveSizeLB = waveSizeLB;
        this.waveSizeUB = waveSizeUB;
    }

    public static class Individual {
        protected ArrayList<Boolean> genome;
        protected double fitness;

        public Individual(ArrayList<Boolean> genome, double fitness) {
            this.genome = genome;
            this.fitness = fitness;
        }

    }


    public ChallengeSolution solve(StopWatch stopWatch) {

        ChallengeSolution greedySolution = constructPureGreedySolution();
        double greedyQuality = computeObjectiveFunction(greedySolution);
        System.out.println("greedy Solution with quality: " + greedyQuality);
        System.out.println("is Feasible: " + isSolutionFeasible(greedySolution));
        ChallengeSolution SASolution = simulatedAnnealing(greedySolution,100,0.80);
        double SAQuality = computeObjectiveFunction(SASolution);
        System.out.println("Simulated Annealing Solution with quality: " + SAQuality);
        System.out.println("is Feasible: " + isSolutionFeasible(SASolution));

        return new ChallengeSolution(new HashSet<>(greedySolution.orders()), new HashSet<>(greedySolution.aisles()));
    }

    private ChallengeSolution constructPureGreedySolution() {
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();

        Map<Integer, Integer> orderItemCount = new HashMap<>();
        for (int i = 0; i < orders.size(); i++) {
            int totalItems = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
            orderItemCount.put(i, totalItems);
        }

        List<Integer> candidateOrders = new ArrayList<>(orderItemCount.keySet());
        candidateOrders.sort((o1, o2) -> Integer.compare(orderItemCount.get(o2), orderItemCount.get(o1)));

        int totalSelectedItems = 0;

        for (int order : candidateOrders) {
            int orderItems = orderItemCount.get(order);

            if (totalSelectedItems + orderItems > waveSizeUB) {
                continue;
            }

            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(selectedOrders, selectedAisles);

            Set<Integer> orderAisles = selectAislesForOrder(order, selectedAisles, itemsLeftInAisles);

            if (orderAisles.isEmpty()) {
                continue;
            }

            selectedOrders.add(order);
            selectedAisles.addAll(orderAisles);
            totalSelectedItems += orderItems;
        }

        return new ChallengeSolution(selectedOrders, selectedAisles);
    }


    public ChallengeSolution simulatedAnnealing(ChallengeSolution initialSolution, double initialTemperature, double coolingRate) {
        Random rand = new Random();

        ChallengeSolution currentSolution = new ChallengeSolution(
            new HashSet<>(initialSolution.orders()),
            new HashSet<>(initialSolution.aisles())
        );
        double currentValue = computeObjectiveFunction(currentSolution);

        ChallengeSolution bestSolution = currentSolution;
        double bestValue = currentValue;

        double temperature = initialTemperature;

        long startTime = System.currentTimeMillis();

        while (System.currentTimeMillis() - startTime < MAX_RUNTIME) {
            ChallengeSolution neighbor = generateNeighbor(currentSolution, rand);

            if (!isSolutionFeasible(neighbor)) continue;

            double neighborValue = computeObjectiveFunction(neighbor);
            double delta = neighborValue - currentValue;

            if (delta > 0) {
                currentSolution = neighbor;
                currentValue = neighborValue;
            } else {
                double acceptanceProb = Math.exp(delta / temperature);
                if (rand.nextDouble() < acceptanceProb) {
                    currentSolution = neighbor;
                    currentValue = neighborValue;
                }
            }

            if (currentValue > bestValue) {
                bestSolution = currentSolution;
                bestValue = currentValue;
            }

            temperature *= coolingRate;
        }

        return bestSolution;
    }


    private ChallengeSolution generateNeighbor(ChallengeSolution solution, Random rand) {
        Set<Integer> newOrders = new HashSet<>(solution.orders());
        Set<Integer> newAisles = new HashSet<>(solution.aisles());

        int moveType = rand.nextInt(4); // 0=troca pedido, 1=add pedido, 2=remove pedido, 3=remove corredor

        switch (moveType) {
            case 0:
                if (!newOrders.isEmpty()) {
                    int orderToRemove = new ArrayList<>(newOrders).get(rand.nextInt(newOrders.size()));
                    newOrders.remove(orderToRemove);

                    int newOrder = rand.nextInt(orders.size());
                    newOrders.add(newOrder);

                    HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(newOrders, newAisles);
                    Set<Integer> orderAisles = selectAislesForOrder(newOrder, newAisles, itemsLeftInAisles);
                    newAisles.addAll(orderAisles);
                }
                break;

            case 1:
                int candidateOrder = rand.nextInt(orders.size());
                newOrders.add(candidateOrder);

                HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(newOrders, newAisles);
                Set<Integer> orderAisles = selectAislesForOrder(candidateOrder, newAisles, itemsLeftInAisles);
                newAisles.addAll(orderAisles);
                break;

            case 2:
                if (!newOrders.isEmpty()) {
                    int orderToRemove = new ArrayList<>(newOrders).get(rand.nextInt(newOrders.size()));
                    newOrders.remove(orderToRemove);

                    newAisles = recalculateAisles(newOrders, newAisles);
                }
                break;

            case 3:
                if (!newAisles.isEmpty()) {
                    int aisleToRemove = new ArrayList<>(newAisles).get(rand.nextInt(newAisles.size()));
                    newAisles.remove(aisleToRemove);

                    ChallengeSolution testSolution = new ChallengeSolution(newOrders, newAisles);
                    if (!isSolutionFeasible(testSolution)) {
                        newAisles.add(aisleToRemove);
                    }
                }
                break;
        }

        return new ChallengeSolution(newOrders, newAisles);
    }

    private Set<Integer> recalculateAisles(Set<Integer> ordersSet, Set<Integer> currentAisles) {
        Set<Integer> newAisles = new HashSet<>();
        HashMap<Integer, Integer> itemsLeftInAisles = new HashMap<>();

        for (int order : ordersSet) {
            Set<Integer> neededAisles = selectAislesForOrder(order, newAisles, itemsLeftInAisles);
            newAisles.addAll(neededAisles);
        }

        return newAisles;
    }


    protected boolean isSolutionFeasible(ChallengeSolution challengeSolution) {
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();
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
            return false;
        }
        if (totalUnits > waveSizeUB) {

            return false;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {

                return false;
            }
        }

        return true;
    }

    double computeObjectiveFunction(ChallengeSolution challengeSolution) {
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
        return (double) totalUnitsPicked / numVisitedAisles;
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

        /* System.out.println("Items left in the aisles: " + itemsLeftInAisles); */

        return new HashMap<>(itemsLeftInAisles);
    }

    private Set<Integer> selectAislesForOrder(int order, Set<Integer> selectedAisles,
        Map<Integer, Integer> itemsLeftInAisles) {
        Map<Integer, Integer> currentOrder = this.orders.get(order);

        Set<Integer> possibleAisles = new HashSet<>(selectedAisles);

        List<Integer> aisleIndexList = IntStream.range(0, aisles.size()).boxed().collect(Collectors.toList());

        Collections.shuffle(aisleIndexList);

        /*
         * System.out.println("Aisle Index order: " + aisleIndexList);
         */
        if (canOrderFitInAisles(currentOrder, itemsLeftInAisles)) {
            /*
             * System.out.println("This order " + currentOrder +
             * "fits in the current aisles: " + itemsLeftInAisles);
             */
            return possibleAisles;
        }

        /* System.out.println("Checking aisles for order " + order); */
        for (int aisle : aisleIndexList) {
            if (!possibleAisles.contains(aisle)) {

                /* System.out.println("Adding aisle " + aisle + " to the possible aisles"); */

                possibleAisles.add(aisle);

                // Adds new offered items to itemsLeftInAisles
                for (Map.Entry<Integer, Integer> entry : this.aisles.get(aisle).entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    itemsLeftInAisles.put(item, itemsLeftInAisles.getOrDefault(item, 0) + quantity);
                }

                /*
                 * System.out.println("Items left in the aisles after adding aisle " + aisle +
                 * ": " + itemsLeftInAisles);
                 */

                if (canOrderFitInAisles(currentOrder, itemsLeftInAisles)) {
                    /*
                     * System.out.println("After adding aisles, this order " + currentOrder +
                     * "fits in the current aisles: " + itemsLeftInAisles);
                     */
                    return possibleAisles;
                }
            }
        }

        /* System.out.println("No more aisles can fit for order " + order); */

        return Collections.emptySet();
    }

    private boolean canOrderFitInAisles(Map<Integer, Integer> newOrder, Map<Integer, Integer> itemsLeftInAisles) {
        for (Map.Entry<Integer, Integer> entry : newOrder.entrySet()) {
            int item = entry.getKey();
            int quantity = entry.getValue();
            if (itemsLeftInAisles.getOrDefault(item, 0) < quantity) {
                /*
                 * System.out.println("Item " + item + " with quantity " + quantity +
                 * " cannot fit in the aisles");
                 */
                return false;
            }
        }

        return true;
    }
}
