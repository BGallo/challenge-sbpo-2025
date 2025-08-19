package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 540000; // milliseconds; 5 s
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

        ChallengeSolution greedySolution = constructGreedySolution();
        double greedyQuality = computeObjectiveFunction(greedySolution);
        System.out.println("greedy Solution with quality: " + greedyQuality);
        System.out.println("is Feasible: " + isSolutionFeasible(greedySolution));


        return new ChallengeSolution(new HashSet<>(greedySolution.orders()), new HashSet<>(greedySolution.aisles()));
    }

    private ChallengeSolution constructGreedySolution() {
        Random rand = new Random();

        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();

        List<Integer> candidateOrders = new ArrayList<>();
        Map<Integer, Integer> orderItemCount = new HashMap<>();

        for (int i = 0; i < orders.size(); i++) {
            int totalItems = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
            orderItemCount.put(i, totalItems);
            candidateOrders.add(i);
        }

        int totalSelectedItems = 0;
        int failedAttempts = 0;
        int maxFailedAttempts = (int) Math.round(candidateOrders.size() * 0.1);

        while (!candidateOrders.isEmpty() && failedAttempts < maxFailedAttempts) {
            int selectedOrder = candidateOrders.get(rand.nextInt(candidateOrders.size()));
            int selectedOrderItems = orderItemCount.get(selectedOrder);

            if (totalSelectedItems + selectedOrderItems > waveSizeUB) {
                candidateOrders.remove(Integer.valueOf(selectedOrder));
                failedAttempts++;
                continue;
            }

            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(selectedOrders, selectedAisles);

            Set<Integer> orderAisles = selectAislesForOrder(selectedOrder, selectedAisles, itemsLeftInAisles);

            if (orderAisles.isEmpty()) {
                candidateOrders.remove(Integer.valueOf(selectedOrder));
                failedAttempts++;
                continue;
            }

            selectedOrders.add(selectedOrder);
            selectedAisles.addAll(orderAisles);
            totalSelectedItems += selectedOrderItems;

            candidateOrders.remove(Integer.valueOf(selectedOrder));
        }

        return new ChallengeSolution(selectedOrders, selectedAisles);
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
