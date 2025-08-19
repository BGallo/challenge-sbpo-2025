package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.util.*;
import java.util.concurrent.TimeUnit;
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

    public double computeObjectiveFunction(ChallengeSolution challengeSolution){
        return 0.0;
    }
}
