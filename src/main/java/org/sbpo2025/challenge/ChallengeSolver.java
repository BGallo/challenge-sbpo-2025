package org.sbpo2025.challenge;
import com.google.ortools.Loader;
import com.google.ortools.sat.*;
import org.apache.commons.lang3.time.StopWatch;

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

    public ChallengeSolution solve(StopWatch stopWatch) {
        Loader.loadNativeLibraries();
        CpModel model = new CpModel();

        int nOrders = orders.size();
        int nAisles = aisles.size();

        // Order selection variables
        IntVar[] orderVars = new IntVar[nOrders];
        for (int i = 0; i < nOrders; i++) {
            orderVars[i] = model.newBoolVar("order_" + i);
        }

        // Constraint: The number of selected orders must be between waveSizeLB and waveSizeUB
        IntVar numOrdersInWave = model.newIntVar(waveSizeLB, waveSizeUB, "num_orders_in_wave");
        model.addEquality(LinearExpr.sum(orderVars), numOrdersInWave);

        // Aisle selection variables
        IntVar[] aisleVars = new IntVar[nAisles];
        for (int j = 0; j < nAisles; j++) {
            aisleVars[j] = model.newBoolVar("aisle_" + j);
        }

        // Ensure an aisle is visited if any order in it is selected
        for (int j = 0; j < nAisles; j++) {
            List<IntVar> relevantOrders = new ArrayList<>();
            for (int i = 0; i < nOrders; i++) {
                if (orders.get(i).containsKey(j)) {
                    relevantOrders.add(orderVars[i]);
                }
            }
            if (!relevantOrders.isEmpty()) {
                model.addGreaterOrEqual(LinearExpr.sum(relevantOrders.toArray(new IntVar[0])), aisleVars[j]);
            }
        }

        // Total units picked variable
        IntVar totalUnitsPicked = model.newIntVar(waveSizeLB, waveSizeUB, "total_units_picked");
        List<IntVar> pickedUnits = new ArrayList<>();

        for (int i = 0; i < nOrders; i++) {
            int orderTotal = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
            if (orderTotal > 0) {
                IntVar picked = model.newIntVar(0, orderTotal, "picked_units_" + i);
                pickedUnits.add(picked);
                model.addMultiplicationEquality(picked, new IntVar[]{orderVars[i], model.newConstant(orderTotal)});
            }
        }
        model.addEquality(LinearExpr.sum(pickedUnits.toArray(new IntVar[0])), totalUnitsPicked);

        // Constraint: Units picked cannot exceed available stock
        for (int item = 0; item < nItems; item++) {
            int finalItem = item;
            int totalAvailable = aisles.stream().mapToInt(a -> a.getOrDefault(finalItem, 0)).sum();
            if (totalAvailable > 0) {
                IntVar totalPicked = model.newIntVar(0, totalAvailable, "total_picked_item_" + item);
                List<IntVar> pickedPerOrder = new ArrayList<>();

                for (int i = 0; i < nOrders; i++) {
                    int pickedAmount = orders.get(i).getOrDefault(item, 0);
                    if (pickedAmount > 0) {
                        IntVar pickedVar = model.newIntVar(0, pickedAmount, "picked_item_" + item + "_order_" + i);
                        pickedPerOrder.add(pickedVar);
                        model.addMultiplicationEquality(pickedVar, new IntVar[]{orderVars[i], model.newConstant(pickedAmount)});
                    }
                }
                model.addEquality(LinearExpr.sum(pickedPerOrder.toArray(new IntVar[0])), totalPicked);
                model.addLessOrEqual(totalPicked, totalAvailable);
            }
        }

        // Ensure feasibility: picked items <= available items in visited aisles
        for (int item = 0; item < nItems; item++) {
            int finalItem = item;
            int totalStock = aisles.stream().mapToInt(a -> a.getOrDefault(finalItem, 0)).sum();
            if (totalStock > 0) {
                IntVar pickedForItem = model.newIntVar(0, totalStock, "picked_for_item_" + item);
                List<IntVar> pickingOrders = new ArrayList<>();

                for (int i = 0; i < nOrders; i++) {
                    int orderQuantity = orders.get(i).getOrDefault(item, 0);
                    if (orderQuantity > 0) {
                        IntVar orderPicked = model.newIntVar(0, orderQuantity, "picked_item_" + item + "_order_" + i);
                        pickingOrders.add(orderPicked);
                        model.addMultiplicationEquality(orderPicked, new IntVar[]{orderVars[i], model.newConstant(orderQuantity)});
                    }
                }
                model.addEquality(LinearExpr.sum(pickingOrders.toArray(new IntVar[0])), pickedForItem);
                model.addLessOrEqual(pickedForItem, totalStock);
            }
        }

        // Minimize the number of aisles visited while maximizing total units picked
        IntVar numVisitedAisles = model.newIntVar(1, nAisles, "num_visited_aisles");
        model.addEquality(LinearExpr.sum(aisleVars), numVisitedAisles);

        int scalingFactor = 100;
        IntVar scaledTotalUnits = model.newIntVar(0, Integer.MAX_VALUE, "scaled_total_units");
        model.addMultiplicationEquality(scaledTotalUnits, new IntVar[]{totalUnitsPicked, model.newConstant(scalingFactor)});
        model.maximize(LinearExpr.sum(new IntVar[]{scaledTotalUnits, model.newConstant(-numVisitedAisles.getIndex())}));

        // Solve the model
        CpSolver solver = new CpSolver();
        solver.getParameters().setMaxTimeInSeconds(600);  // Set a time limit
        CpSolverStatus status = solver.solve(model);

        // Extract solution if feasible
        if (status == CpSolverStatus.OPTIMAL || status == CpSolverStatus.FEASIBLE) {
            Set<Integer> selectedOrders = new HashSet<>();
            Set<Integer> selectedAisles = new HashSet<>();
            for (int i = 0; i < nOrders; i++) {
                if (solver.value(orderVars[i]) == 1) {
                    selectedOrders.add(i);
                }
            }
            for (int j = 0; j < nAisles; j++) {
                if (solver.value(aisleVars[j]) == 1) {
                    selectedAisles.add(j);
                }
            }
            return new ChallengeSolution(selectedOrders, selectedAisles);
        } else {
            return null;  // No feasible solution found
        }
    }

    public ArrayList<Individual> generateInitialPopulation(int size) {
        ArrayList<Individual> population = new ArrayList<>();
        Random rand = new Random();

        List<Integer> orderIndexes = new ArrayList<>();
        for (int i = 0; i < orders.size(); i++) {
            orderIndexes.add(i);
        }


        for (int i = 0; i < size; i++) {
            ArrayList<Boolean> genome = new ArrayList<>(Collections.nCopies(orders.size() + aisles.size(), false));
            int pickedUnits = 0;
            List<Integer> localOrderIndexes = new ArrayList<>(orderIndexes);
            List<Integer> localAisleIndexes = new ArrayList<>();
            List<Map<Integer, Integer>> unitsTakenFromAisles = new ArrayList<>(Collections.nCopies(aisles.size(), new HashMap<>()));


            while (!localOrderIndexes.isEmpty()) {
                int orderIndex = rand.nextInt(localOrderIndexes.size());
                int pickedOrder = localOrderIndexes.get(orderIndex);
                int orderUnitsSum = orders.get(pickedOrder).values().stream().mapToInt(Integer::intValue).sum();

                localOrderIndexes.remove(orderIndex);

                if (pickedUnits + orderUnitsSum > waveSizeUB) continue;

                pickedUnits += orderUnitsSum;
                genome.set(pickedOrder, true);

                for(Map.Entry<Integer, Integer> entry: orders.get(pickedOrder).entrySet()){
                    int item = entry.getKey();
                    int quantity = entry.getValue();

                    for(int aisleIndex = 0; aisleIndex < this.aisles.size(); aisleIndex++){
                        Map<Integer, Integer> aisle = this.aisles.get(aisleIndex);
                        if(aisle.containsKey(item)){
                            int amountTakenFromItemInAisle = unitsTakenFromAisles.get(aisleIndex).getOrDefault(item, 0);
                            int amountAvailableInAisle = aisle.get(item);
                            if (amountTakenFromItemInAisle == amountAvailableInAisle) continue;
                            if (!localAisleIndexes.contains(aisleIndex)) localAisleIndexes.add(aisleIndex);
                            int newItemInAisleAmount = Math.min(amountAvailableInAisle, amountTakenFromItemInAisle + quantity);
                            unitsTakenFromAisles.get(aisleIndex).put(item, newItemInAisleAmount);
                            genome.set(orders.size() + aisleIndex, true);
                            if (amountTakenFromItemInAisle + quantity < amountAvailableInAisle) break;
                            quantity -= amountAvailableInAisle - amountTakenFromItemInAisle;
                        }
                    }

                }
            }
            population.add(new Individual(genome, -1));
        }
        return population;
    }

    public ChallengeSolution decodeIndividual(Individual individual) {
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();
        
        for (int i = 0; i < orders.size(); i++) {
            if (individual.genome.get(i)) {
                selectedOrders.add(i);
            }
        }
        
        for (int i = 0; i < aisles.size(); i++) {
            if (individual.genome.get(orders.size() + i)) {
                selectedAisles.add(i);
            }
        }
        
        return new ChallengeSolution(selectedOrders, selectedAisles);
    }

    //One point crossover
    public Individual crossover(Individual parent1, Individual parent2) {
        Random rand = new Random();
        int crossoverPoint = rand.nextInt(parent1.genome.size());
        ArrayList<Boolean> childGenome = new ArrayList<>();
        for (int i = 0; i < parent1.genome.size(); i++) {
            if (i < crossoverPoint) {
                childGenome.add(parent1.genome.get(i));
            } else {
                childGenome.add(parent2.genome.get(i));
            }
        }
        Individual child = new Individual(childGenome, -1);
        mutate(child);
        return child;
    }


    public void mutate(Individual individual) {
        Random rand = new Random();
        if (rand.nextDouble() > 0.1) return;
        int mutationPoint = rand.nextInt(individual.genome.size());
        individual.genome.set(mutationPoint, !individual.genome.get(mutationPoint));
    }

    /*
     * Recieves an already sorted population
     * Returns a population of size n * 1.5
     */
    public ArrayList<Individual> crossOverPopulation(ArrayList<Individual> population) {
        ArrayList<Individual> newPopulation = new ArrayList<>();
        Random rand = new Random();
        while(!population.isEmpty()){
            Individual parent1 = population.remove(0);
            Individual parent2;
            newPopulation.add(parent1);
            //biased for loop
            if (population.isEmpty()) break;
            for (int i = 0; i < population.size(); i++) {
                if (rand.nextDouble() < 0.5 || i == population.size() - 1) {
                    parent2 = population.remove(i);
                    newPopulation.add(parent2);
                    newPopulation.add(crossover(parent1, parent2));
                }
            }
        }
        return newPopulation;
        
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

        int penalty = 0;
        if (!isSolutionFeasible(challengeSolution)) {
            penalty -= 10;
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
}
