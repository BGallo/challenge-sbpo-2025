package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 30000; // milliseconds; 30 s
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
        double alpha = 0.80;

        long startTime = System.currentTimeMillis();

        RVNDSolution bestSolution = constructGreedyRandomizedSolution(alpha);
        double bestQuality = computeObjectiveFunction(bestSolution);

        System.out.println("Initial Solution: " + bestSolution + " with quality: " + bestQuality);
        System.out.println("is Feaseble: " + isSolutionFeasible(bestSolution, true));

        int currentIteration = 0;
        int maxIterations = Integer.MAX_VALUE;
        while (System.currentTimeMillis() - startTime < MAX_RUNTIME && currentIteration < maxIterations) {
            currentIteration++;

            RVNDSolution currentSolution = constructGreedyRandomizedSolution(alpha);
            double currentQuality = bestQuality;

            currentSolution = randomVariableNeighborhoodDescent(currentSolution, currentQuality);
            currentQuality = computeObjectiveFunction(currentSolution);

            if (currentQuality > bestQuality) {
                bestQuality = currentQuality;
                bestSolution = currentSolution;
                System.out.println("New best solution found in Iteration " + currentIteration + ": " + bestQuality);
            }
        }

         
        double graspFitness = 0.0;

        if(bestSolution != null) {
            graspFitness = computeObjectiveFunction(bestSolution);
            System.out.println("Best Score GRASP:" + graspFitness);
            System.out.println("É viável: " + isSolutionFeasible(bestSolution, true));
            System.out.println("Solution: " + bestSolution);
        }

        return new ChallengeSolution(new HashSet<>(bestSolution.orders()), new HashSet<>(bestSolution.aisles()));
    }

    
    
    private RVNDSolution constructGreedyRandomizedSolution(double alpha) {
        Random rand = new Random();

        //base responses
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();

        //the index of the candidates
        List<Integer> candidateOrders = new ArrayList<>();

        //variables that determine the number of items in the order
        Map<Integer, Integer> orderItemCount = new HashMap<>();

        for (int i = 0; i < orders.size(); i++) {
            int totalItems = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
            orderItemCount.put(i, totalItems);
            candidateOrders.add(i);
        }

        int totalSelectedItems = 0;

        while (!candidateOrders.isEmpty()) {
            List<Integer> RCL = buildRCL(candidateOrders, orderItemCount, alpha);

            if (RCL.isEmpty()) break;
            /* System.out.println(RCL); */

            int selectedOrder = RCL.get(rand.nextInt(RCL.size()));
            int selectedOrderItems = orderItemCount.get(selectedOrder);

            /* System.out.println("The order selected is " + selectedOrder + " with " + selectedOrderItems + " items"); */

            if (totalSelectedItems + selectedOrderItems > waveSizeUB) {
                candidateOrders.remove(Integer.valueOf(selectedOrder));
                /* System.out.println("The order was removed because it exceeded the wave size UB"); */
                continue;
            }

            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(selectedOrders, selectedAisles);

            Set<Integer> orderAisles = selectAislesForOrder(selectedOrder, selectedAisles, itemsLeftInAisles);
            
            if (orderAisles.isEmpty()) {
                candidateOrders.remove(Integer.valueOf(selectedOrder));
                continue;
            }

            selectedOrders.add(selectedOrder);
            selectedAisles.addAll(orderAisles);
            
            totalSelectedItems += selectedOrderItems;

            candidateOrders.remove(Integer.valueOf(selectedOrder));
        }

        return new RVNDSolution(new ArrayList<>(selectedOrders), new ArrayList<>(selectedAisles));
    }

    private HashMap<Integer, Integer> getItemsLeftInAisles(Set<Integer> selectedOrders, Set<Integer> selectedAisles) {
        HashMap<Integer, Integer> itemsLeftInAisles = new HashMap<>();

        //Adds all aisles items to the itemsLeftInAisles
        for (int aisle : selectedAisles) {
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                int item = entry.getKey();
                int quantity = entry.getValue();
                itemsLeftInAisles.put(item, itemsLeftInAisles.getOrDefault(item, 0) + quantity);
            }
        }

        //Subtracts the items that were selected in the orders
        for (int order : selectedOrders) {
            for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                int item = entry.getKey();
                int quantity = entry.getValue();
                itemsLeftInAisles.put(item, itemsLeftInAisles.getOrDefault(item, 0) - quantity);
            }
        }

        /* System.out.println("Items left in the aisles: " + itemsLeftInAisles); */

        return itemsLeftInAisles;
    }

    private List<Integer> buildRCL(List<Integer> candidateOrders, Map<Integer, Integer> orderItemCount, double alpha) {
        List<Integer> RCL = new ArrayList<>();
        int maxItems = candidateOrders.stream().mapToInt(orderItemCount::get).max().orElse(0);
        int minItems = candidateOrders.stream().mapToInt(orderItemCount::get).min().orElse(0);
        double threshold = maxItems - alpha * (maxItems - minItems);

        for (int order : candidateOrders) {
            if (orderItemCount.get(order) >= threshold) {
                RCL.add(order);
            }
        }
        return RCL;
    }

    private Set<Integer> selectAislesForOrder(int order, Set<Integer> selectedAisles, Map<Integer, Integer> itemsLeftInAisles) {
        Map<Integer, Integer> currentOrder = this.orders.get(order);

        Set<Integer> possibleAisles = new HashSet<>(selectedAisles);

        List<Integer> aisleIndexList = IntStream.range(0, aisles.size()).boxed().collect(Collectors.toList());

        Collections.shuffle(aisleIndexList);

        /* System.out.println("Aisle Index order: " + aisleIndexList);
 */
        if (canOrderFitInAisles(currentOrder, itemsLeftInAisles)) {
           /*  System.out.println("This order " + currentOrder + "fits in the current aisles: " + itemsLeftInAisles); */
            return possibleAisles;
        }

        /* System.out.println("Checking aisles for order " + order); */
        for (int aisle : aisleIndexList) {
            if (!possibleAisles.contains(aisle)) {

                /* System.out.println("Adding aisle " + aisle + " to the possible aisles"); */

                possibleAisles.add(aisle);
                
                //Adds new offered items to itemsLeftInAisles
                for (Map.Entry<Integer,Integer> entry : this.aisles.get(aisle).entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    itemsLeftInAisles.put(item, itemsLeftInAisles.getOrDefault(item, 0) + quantity);
                }

                /* System.out.println("Items left in the aisles after adding aisle " + aisle + ": " + itemsLeftInAisles); */

                if (canOrderFitInAisles(currentOrder, itemsLeftInAisles)) {
                    /* System.out.println("After adding aisles, this order " + currentOrder + "fits in the current aisles: " + itemsLeftInAisles); */
                    return possibleAisles;
                }
            }
        }

        /* System.out.println("No more aisles can fit for order " + order); */

        return Collections.emptySet();
    }

    private boolean canOrderFitInAisles(Map<Integer, Integer> newOrder, Map<Integer, Integer> itemsLeftInAisles) {
        
        for(Map.Entry<Integer,Integer> entry : newOrder.entrySet()) {
            int item = entry.getKey();
            int quantity = entry.getValue();
            if(itemsLeftInAisles.getOrDefault(item,0) < quantity) {
                /* System.out.println("Item " + item + " with quantity " + quantity + " cannot fit in the aisles"); */
                return false;
            }
        }

        return true;
    }

    class Neighborhood {
        int iterator;

        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality) {
            return null;
        }

        boolean move(RVNDSolution solution) { return false; }

        boolean hasNext(RVNDSolution solution) {
            return true;
        }

        void start() {
            iterator = 0;
            /* System.out.println("Starting Neighborhood"); */
        }

        void next() {
            iterator++;
        }

    }

    class RemoveAisle extends Neighborhood{

        @Override
        boolean hasNext(RVNDSolution solution) {
            /* System.out.println("Checking if iteration is over: " + iterator + " < " + solution.aisles().size()); */
            return iterator < solution.aisles().size() && iterator < 100;
        }

        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality) {
            /* System.out.println("Exploring Remove Aisle Neighborhood"); */

            RVNDSolution bestSolution = initialSolution;
            double bestQuality = initialQuality;

            /* System.out.println("Initial Solution: " + bestSolution + " with quality: " + bestQuality); */

            Collections.shuffle(initialSolution.aisles());

            this.start();
            while (this.hasNext(initialSolution)) {
                RVNDSolution currentSolution = new RVNDSolution(new ArrayList<>(initialSolution.orders()), new ArrayList<>(initialSolution.aisles()));

                boolean isViable = this.move(currentSolution);

                if (!isViable) {
                    this.next();
                    continue;
                }

                double currentQuality = computeObjectiveFunction(currentSolution);

                /* System.out.println("Current Solution: " + currentSolution + " with quality: " + currentQuality);
                System.out.println("Best Solution: " + bestSolution + " with quality: " + bestQuality); */
                if (currentQuality > bestQuality) {
                    bestQuality = currentQuality;
                    bestSolution = currentSolution;
                    /* System.out.println("New best solution found: " + bestQuality); */
                }
                
                this.next();
            }

            /* System.out.println("Best Solution after Remove Aisle Neighborhood: " + bestSolution + " with quality: " + bestQuality); */
            return bestSolution;
        }

        //A move configures a removal of an aisle and, from there, the removal of any other aisle that can be removed
        @Override
        boolean move(RVNDSolution solution) {
            /* System.out.println("Moving Remove Aisle Neighborhood in solution: " + solution + " with iterator: " + iterator); */
            solution.aisles().remove(iterator);

            if(!isSolutionFeasible(solution, false)) return false;

            int nextRemovableAisle = getNextRemovableAisle(solution);
            while (nextRemovableAisle != -1) {
                solution.aisles().remove(nextRemovableAisle);
                nextRemovableAisle = getNextRemovableAisle(solution);
            }

            return true;

            /* System.out.println("Solution after move: " + solution); */
        }

        int getNextRemovableAisle(RVNDSolution solution) {
            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(new HashSet<>(solution.orders()), new HashSet<>(solution.aisles()));

            for(int i = 0; i < solution.aisles().size(); i++) {

                int aisleIndex = solution.aisles().get(i);
                HashMap<Integer, Integer> aisleItems = new HashMap<>(aisles.get(aisleIndex));
                boolean canRemove = true;

                for(Map.Entry<Integer,Integer> entry : aisleItems.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    if(itemsLeftInAisles.getOrDefault(item,0) < quantity) {
                        canRemove = false;
                        break;
                    }
                }

                if(canRemove) return i;
            }
            return -1;
        }
    }

    class AddOrder extends Neighborhood {
        ArrayList<Integer> candidateOrders;

        @Override
        boolean hasNext(RVNDSolution solution) {
            return iterator < (orders.size() - solution.orders().size()) && iterator < 100;
        }
        
        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality) {
            /* System.out.println("Exploring Add Order Neighborhood"); */

            RVNDSolution bestSolution = initialSolution;
            double bestQuality = initialQuality;

            this.candidateOrders = IntStream.range(0, orders.size()).boxed().filter((order) -> !initialSolution.orders().contains(order)).collect(Collectors.toCollection(ArrayList::new));
            Collections.shuffle(candidateOrders);

            /* System.out.println("Candidate Orders: " + candidateOrders);

            System.out.println("Initial Solution: " + bestSolution + " with quality: " + bestQuality); */

            this.start();
            while (this.hasNext(initialSolution)) {
                RVNDSolution currentSolution = new RVNDSolution(new ArrayList<>(initialSolution.orders()), new ArrayList<>(initialSolution.aisles()));

                boolean isViable = this.move(currentSolution);

                if (!isViable) {
                    this.next();
                    continue;
                }

                double currentQuality = computeObjectiveFunction(currentSolution);

                if (currentQuality > bestQuality) {
                    bestQuality = currentQuality;
                    bestSolution = currentSolution;
                }

                this.next();
            }

            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution) {
            /* System.out.println("Moving Add Order Neighborhood in solution: " + solution + " with iterator: " + iterator); */
            solution.orders().add(candidateOrders.get(iterator));

            if(!isSolutionFeasible(solution, false)) return false;

            int nextCandidateOrder = getNextCandidateOrderToAdd(solution);
            while (nextCandidateOrder != -1) {
                solution.orders().add(candidateOrders.get(nextCandidateOrder));
                nextCandidateOrder = getNextCandidateOrderToAdd(solution);
            }

            System.out.println("Solution after move: " + solution + "with iterator: " + iterator);

            return true;
        }

        int getNextCandidateOrderToAdd(RVNDSolution solution) {
            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(new HashSet<>(solution.orders()), new HashSet<>(solution.aisles()));

            for(int i = 0; i < candidateOrders.size(); i++) {
                int orderIndex = candidateOrders.get(i);
                HashMap<Integer, Integer> orderItems = new HashMap<>(orders.get(orderIndex));
                boolean canAdd = true;

                for(Map.Entry<Integer,Integer> entry : orderItems.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    if(itemsLeftInAisles.getOrDefault(item,0) < quantity) {
                        canAdd = false;
                        break;
                    }
                }

                if(canAdd) return i;
            }

            return -1;
        }
    }

    private RVNDSolution randomVariableNeighborhoodDescent(RVNDSolution solution, double quality) {
        RVNDSolution bestSolution = solution;
        double bestQuality = quality;

        ArrayList<Neighborhood> neighborhoods = new ArrayList<>();
        neighborhoods.add(new RemoveAisle());
        neighborhoods.add(new AddOrder());

        Collections.shuffle(neighborhoods);

        for (Neighborhood neighborhood : neighborhoods) {
            RVNDSolution currentSolution = neighborhood.explore(bestSolution, bestQuality);
            double currentQuality = computeObjectiveFunction(currentSolution);

            if (currentQuality > bestQuality) {
                bestQuality = currentQuality;
                bestSolution = currentSolution;
            }
        }

        return bestSolution;
    }

    public ChallengeSolution decodeIndividual(Individual individual) {
        ArrayList<Boolean> genome = individual.genome;
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();

        for (int i = 0; i < orders.size(); i++) {
            if (genome.get(i)) {
                selectedOrders.add(i);
            }
        }

        for (int i = orders.size(); i < genome.size(); i++) {
            if (genome.get(i)) {
                selectedAisles.add(i - orders.size());
            }
        }

        return new ChallengeSolution(selectedOrders, selectedAisles);
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

    protected double computeObjectiveFunction(RVNDSolution challengeSolution) {
        List<Integer> selectedOrders = challengeSolution.orders();
        List<Integer> visitedAisles = challengeSolution.aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return 0.0;
        }

        int penalty = 0;
        if (!isSolutionFeasible(challengeSolution, false)) {
            penalty -= applyPenalty(challengeSolution);
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
        return (double) (totalUnitsPicked / numVisitedAisles) + penalty;
    }

    protected int applyPenalty(RVNDSolution challengeSolution) {
        int penalty = 0;
        List<Integer> selectedOrders = challengeSolution.orders();
        List<Integer> visitedAisles = challengeSolution.aisles();

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
            penalty += (waveSizeLB - totalUnits) * nItems;
        }

        if (totalUnits > waveSizeUB) {
            penalty += (totalUnits - waveSizeLB) * nItems;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {
                penalty += (totalUnitsPicked[i] - totalUnitsAvailable[i]) * nItems;
            }
        }

        return penalty;

    }

    public enum PopulationSelectionType {
        ELITIST, TOURNAMENT
    }

    public static void chooseNextGeneration(ArrayList<Individual> population, int populationSize, PopulationSelectionType selectionType) {
        if (selectionType == PopulationSelectionType.ELITIST) {
            population.subList(populationSize, population.size()).clear();
        } else if (selectionType == PopulationSelectionType.TOURNAMENT) {
            Random rand = new Random();
            while (population.size() > populationSize) {
                int index1 = rand.nextInt(population.size());
                int index2 = rand.nextInt(population.size());
                if (population.get(index1).fitness > population.get(index2).fitness) {
                    population.remove(index2);
                } else {
                    population.remove(index1);
                }
            }
        }
    }
}
