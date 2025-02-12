package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.sql.SQLOutput;
import java.util.*;
import java.util.concurrent.TimeUnit;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 600000; // milliseconds; 10 minutes

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

    public ChallengeSolution solve(StopWatch stopWatch) {
        ArrayList<ArrayList<Boolean>> population = this.generateInitialPopulation(10);
        for(ArrayList<Boolean> individual: population){
            System.out.println(individual);
        }
        return null;
    }

    public ArrayList<ArrayList<Boolean>> generateInitialPopulation(int size){

        /*ArrayList<ArrayList<boolean>> population;
        Map<Integer,Integer> itemSums = new Map<>();
        int index = 0;
        for(Map<Integer,Integer> order:this.orders){
            int total = order.values().stream().mapToInt(Integer::intValue).sum();
            itemSums.put(index++,total);
        }
        for(int i = 0; i < size; i++){
            int waveSize = 0;
            List<Map.Entry<K,V>> sorted = itemSums.entrySet().stream().sorted(Map.Entry.comparingByValue()).collect(Collectors.toList());

            ArrayList<boolean> individual;
            while (waveSize < this.waveSizeUB){
                Random random = new Random();
                if(random.nextInt()){
                    int order = random.nextInt(sorted.lenght);
                    int selectedOrderIndex = sorted[order].getKey();
                    Map<Integer, Integer> selectedOrder = this.orders[selectedOrderIndex];
                }
            }
        }*/

        ArrayList<ArrayList<Boolean>> population = new ArrayList<>();
        Random rand = new Random();

        for (int i = 0; i < size; i++) {
            ArrayList<Boolean> individual = new ArrayList<>(Collections.nCopies(orders.size() + aisles.size(), false));
            Set<Integer> selectedOrders = new HashSet<>();
            Set<Integer> selectedAisles = new HashSet<>();
            Map<Integer, Integer> itemAvailability = new HashMap<>();
            int totalItems = 0;
            boolean foundItem = true;
            int oldTotalItems = 0;
            System.out.println("i:"+i);
            while (totalItems < waveSizeUB && foundItem) {
                int orderIndex = rand.nextInt(orders.size());
                if (selectedOrders.contains(orderIndex)) continue;

                Map<Integer, Integer> order = orders.get(orderIndex);
                int orderItems = order.values().stream().mapToInt(Integer::intValue).sum();

                if (totalItems + orderItems > waveSizeUB) continue;

                selectedOrders.add(orderIndex);
                individual.set(orderIndex, true);
                oldTotalItems=totalItems;
                totalItems += orderItems;
                System.out.println(totalItems);

                if(totalItems==oldTotalItems){
                    System.out.println("chegou aqui");
                    foundItem = false;
                    break;
                };
                System.out.println(foundItem);
                for (Map.Entry<Integer, Integer> entry : order.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    int remainingNeeded = quantity - itemAvailability.getOrDefault(item, 0);

                    if (remainingNeeded > 0) {
                        for (int aisleIndex = 0; aisleIndex < aisles.size(); aisleIndex++) {
                            if (selectedAisles.contains(aisleIndex)) continue;

                            Map<Integer, Integer> aisle = aisles.get(aisleIndex);
                            if (aisle.containsKey(item)) {
                                selectedAisles.add(aisleIndex);
                                individual.set(orders.size() + aisleIndex, true);
                                int aisleQuantity = aisle.get(item);
                                itemAvailability.put(item, itemAvailability.getOrDefault(item, 0) + aisleQuantity);

                                if (itemAvailability.get(item) >= quantity) break;
                            }
                        }
                    }
                }
            }
            population.add(individual);
        }
        return population;
    }

    /*
     * Get the remaining time in seconds
     */
    protected long getRemainingTime(StopWatch stopWatch) {
        return Math.max(
                TimeUnit.SECONDS.convert(MAX_RUNTIME - stopWatch.getTime(TimeUnit.MILLISECONDS), TimeUnit.MILLISECONDS),
                0);
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
        if (totalUnits < waveSizeLB || totalUnits > waveSizeUB) {
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
        return (double) totalUnitsPicked / numVisitedAisles;
    }
}
