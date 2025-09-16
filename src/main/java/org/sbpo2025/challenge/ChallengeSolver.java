package org.sbpo2025.challenge;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.time.StopWatch;

import com.gurobi.gurobi.GRB;
import com.gurobi.gurobi.GRBEnv;
import com.gurobi.gurobi.GRBException;
import com.gurobi.gurobi.GRBLinExpr;
import com.gurobi.gurobi.GRBModel;
import com.gurobi.gurobi.GRBVar;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 300; // seconds

    protected List<Map<Integer, Integer>> orders;
    protected List<Map<Integer, Integer>> aisles;
    protected int nItems;
    protected int waveSizeLB;
    protected int waveSizeUB;

    protected double maxPercentage;
    protected double minPercentage;
    protected double currentPercentage;

    protected int maxNoImprovementIterations;

    protected double randomFactor;

    protected HashMap<Neighborhood, Double> weightedNeighborhoods = new HashMap<>() {{
            put(new AddBestAisles(), 1.0);
            put(new RemoveRandomAisles(), 1.0);
            put(new RemoveWorstAisles(), 1.0);
            put(new AddBestOrders(), 1.0);
            put(new RemoveRandomOrders(), 1.0);
            put(new RemoveWorstOrders(), 1.0);
    }};

    protected List<HashMap<Neighborhood, Integer>> bondingFactors = new ArrayList<>();

    protected List<Integer> bestOrdersByItemNumber = new ArrayList<>();
    protected List<Integer> bestAislesByItemNumber = new ArrayList<>();

     public ChallengeSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB, int waveSizeUB, double maxPercentage, double minPercentage, int maxNoImprovementIterations, double randomFactor) {
        this.orders = orders;
        this.aisles = aisles;
        this.nItems = nItems;
        this.waveSizeLB = waveSizeLB;
        this.waveSizeUB = waveSizeUB;
        this.maxPercentage = maxPercentage;
        this.minPercentage = minPercentage;
        this.currentPercentage = minPercentage;
        this.maxNoImprovementIterations = maxNoImprovementIterations;
        this.randomFactor = randomFactor;

        for (int i = 0; i < 6; i++) {
            bondingFactors.add(new HashMap<>());
        }

        bestOrdersByItemNumber = IntStream.range(0, orders.size())
            .boxed()
            .sorted((i, j) -> {
                int sumI = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
                int sumJ = orders.get(j).values().stream().mapToInt(Integer::intValue).sum();
                return Integer.compare(sumJ, sumI);
            })
            .collect(Collectors.toList());

        bestAislesByItemNumber = IntStream.range(0, aisles.size())
            .boxed()
            .sorted((i, j) -> {
                int sumI = aisles.get(i).values().stream().mapToInt(Integer::intValue).sum();
                int sumJ = aisles.get(j).values().stream().mapToInt(Integer::intValue).sum();
                return Integer.compare(sumJ, sumI);
            })
            .collect(Collectors.toList());
    }

    class ALNSSolution {
        List<Integer> selectedAisles = new ArrayList<>();
        List<Integer> selectedOrders = new ArrayList<>();
        Map<Integer, Integer> itensLeftInAisles = new HashMap<>();
        double objectiveValue;
        int totalItemsPicked;

        public ALNSSolution() {
            this.objectiveValue = 0;
        }

        public ALNSSolution(List<Integer> selectedOrders, List<Integer> selectedAisles) {
            this.selectedOrders = selectedOrders;
            this.selectedAisles = selectedAisles;
        }

        public ALNSSolution(ALNSSolution other) {
            this.selectedAisles = new ArrayList<>(other.selectedAisles);
            this.selectedOrders = new ArrayList<>(other.selectedOrders);
            this.objectiveValue = other.objectiveValue;
            this.itensLeftInAisles = new HashMap<>(other.itensLeftInAisles);
            this.totalItemsPicked = other.totalItemsPicked;
        }

        public void calcItemsLeftInAisles() {
            itensLeftInAisles.clear();

            for (int aisle : selectedAisles) {
                for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                    itensLeftInAisles.merge(entry.getKey(), entry.getValue(), Integer::sum);
                }
            }

            for (int order : selectedOrders) {
                for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                    itensLeftInAisles.merge(entry.getKey(), -entry.getValue(), Integer::sum);
                }
            }

            totalItemsPicked = selectedOrders.stream()
                    .mapToInt(o -> orders.get(o).values().stream().mapToInt(Integer::intValue).sum())
                    .sum();
        }

        public String toString() {
            return "Orders: " + selectedOrders.toString() + ", Aisles: " + selectedAisles.toString() + ", Obj: " + objectiveValue;
        }
    }

    public ChallengeSolution solve(StopWatch stopWatch, int nThreads) throws Exception {
        GRBEnv env = initializeGrbEnv(stopWatch, nThreads);
        if (env == null) {
            return null;
        }


        ALNSSolution solution = getInitialViableSolution();
        System.out.println("Initial solution: " + solution.objectiveValue);

        try{
            //1. Initializing GRB Model
            GRBModel model = new GRBModel(env);

            // Variáveis x_p: 1 se pedido p for selecionado
            GRBVar[] x = new GRBVar[orders.size()];
            for (int p = 0; p < orders.size(); p++) {
                x[p] = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "x_" + p);
            }

            // Variáveis y_c: 1 se corredor c for selecionado
            GRBVar[] y = new GRBVar[aisles.size()];
            for (int c = 0; c < aisles.size(); c++) {
                y[c] = model.addVar(0.0, 1.0, 0.0, GRB.BINARY, "y_" + c);
            }

            //Restrição quanto ao máximo e mínimo de itens
            GRBLinExpr totalItemsExpr = new GRBLinExpr();

            for (int p = 0; p < orders.size(); p++) {
                int totalItemsInOrder = orders.get(p).values().stream().mapToInt(Integer::intValue).sum();

                totalItemsExpr.addTerm(totalItemsInOrder, x[p]);
            }

            model.addRange(totalItemsExpr, waveSizeLB, waveSizeUB, "item_bounds");

            //Restrição quanto a oferta e demanda
            for (int i = 0; i < nItems; i++) {
                GRBLinExpr demandExpr = new GRBLinExpr();
                GRBLinExpr supplyExpr = new GRBLinExpr();

                // soma da demanda dos pedidos para o item i
                for (int p = 0; p < orders.size(); p++) {
                    Integer n_pi = orders.get(p).get(i);
                    if (n_pi != null) {
                        demandExpr.addTerm(n_pi, x[p]);
                    }
                }

                // soma da oferta dos corredores para o item i
                for (int c = 0; c < aisles.size(); c++) {
                    Integer n_ci = aisles.get(c).get(i);
                    if (n_ci != null) { 
                        supplyExpr.addTerm(n_ci, y[c]);
                    }
                }

                // adiciona a restrição: demanda <= oferta
                model.addConstr(demandExpr, GRB.LESS_EQUAL, supplyExpr, "cover_item_" + i);
            }

            while (MAX_RUNTIME - stopWatch.getTime(TimeUnit.SECONDS) > 1) {

                solution = exploreWithALNS(solution, stopWatch, nThreads);

                System.out.println("After ALNS: " + solution.objectiveValue + " In " + stopWatch.getTime(TimeUnit.SECONDS) + "s");

                env.set(GRB.DoubleParam.TimeLimit, Math.min(MAX_RUNTIME - stopWatch.getTime(TimeUnit.SECONDS), 60.0));

                solution.objectiveValue = computeObjectiveFunction(solution);

                solution = intensifyWithDinkelbach(model, x, y, solution, stopWatch, env);

                solution.calcItemsLeftInAisles();

                System.out.println("After Dinkelbach: " + solution.objectiveValue + " In " + stopWatch.getTime(TimeUnit.SECONDS) + "s");

            }

            model.dispose();

        } finally {
            env.dispose();
        }

        System.out.println("Is Feasible? " + isSolutionFeasible(solution));
        LoggerUtils.appendColumnsToLine("results.log", solution.objectiveValue);

        return new ChallengeSolution(new HashSet<>(solution.selectedOrders), new HashSet<>(solution.selectedAisles));
    }

    public ALNSSolution intensifyWithDinkelbach(GRBModel model, GRBVar[] x, GRBVar[] y, ALNSSolution solution, StopWatch stopWatch, GRBEnv env) throws GRBException {
        double startTime = stopWatch.getTime(TimeUnit.SECONDS);

        while (stopWatch.getTime(TimeUnit.SECONDS) - startTime < 60 && MAX_RUNTIME - stopWatch.getTime(TimeUnit.SECONDS) > 1) {

            Set<Integer> processedOrders = new HashSet<>(solution.selectedOrders);
            for (int p = 0; p < orders.size(); p++) {
                x[p].set(GRB.DoubleAttr.Start, processedOrders.contains(p) ? 1.0 : 0.0);
            }

            Set<Integer> processedAisles = new HashSet<>(solution.selectedAisles);
            for (int c = 0; c < aisles.size(); c++) {
                y[c].set(GRB.DoubleAttr.Start, processedAisles.contains(c) ? 1.0 : 0.0);
            }

            //Parte de cima da fração
            GRBLinExpr obj = new GRBLinExpr();
            for (int p = 0; p < orders.size(); p++) {
                int totalItemsInOrder = orders.get(p).values().stream().mapToInt(Integer::intValue).sum();
                obj.addTerm(totalItemsInOrder, x[p]);
            }

            //Parte de baixo da fração
            for (int c = 0; c < aisles.size(); c++) {
                obj.addTerm(-solution.objectiveValue, y[c]);
            }

            model.setObjective(obj, GRB.MAXIMIZE);

            env.set(GRB.DoubleParam.TimeLimit, Math.max(60.0 - (stopWatch.getTime(TimeUnit.SECONDS) - startTime), 0.0));

            model.optimize();

            if (model.get(GRB.IntAttr.SolCount) > 0) {

                solution = convertToChallengeSolution(x, y);

                solution.objectiveValue = computeObjectiveFunction(solution);

                if (model.get(GRB.DoubleAttr.ObjVal) <= 1e-6) {
                    break;
                }

            }
        }

        return solution;
    }

    public ALNSSolution exploreWithALNS(ALNSSolution bestSolution, StopWatch stopWatch, int nThreads) throws Exception {
        Random rng = new Random();

        ALNSSolution currentSolution = new ALNSSolution(bestSolution);
        ALNSSolution tempSolution;
        int noImprovementIterations = 0;
        Neighborhood lastNeighborhood = null;

        ExecutorService executor = Executors.newFixedThreadPool(nThreads);

        double startTime = stopWatch.getTime(TimeUnit.SECONDS);

        while (stopWatch.getTime(TimeUnit.SECONDS) - startTime < 60 && MAX_RUNTIME - stopWatch.getTime(TimeUnit.SECONDS) > 1) {
            noImprovementIterations++;

            tempSolution = new ALNSSolution(currentSolution);

            List<Neighborhood> neighborhoodList = new ArrayList<>();

            for (int t = 0; t < nThreads; t++) {
                neighborhoodList.add(selectNeighborhood(weightedNeighborhoods, lastNeighborhood));
            }

            List<Future<ALNSSolution>> futures = new ArrayList<>();

            for (Neighborhood neigh : neighborhoodList) {
                ALNSSolution solutionCopy = new ALNSSolution(currentSolution); // cópia independente
                futures.add(executor.submit(() -> {
                    neigh.move(solutionCopy, currentPercentage, randomFactor);
                    return solutionCopy;
                }));
            }

            ALNSSolution bestMoveCandidate = null;
            Neighborhood bestNeighborhood = null;
            double bestMoveSolution = Double.NEGATIVE_INFINITY;

            for (int i = 0; i < futures.size(); i++) {
                ALNSSolution candidate = futures.get(i).get();
                if (candidate.objectiveValue > bestMoveSolution) {
                    bestMoveSolution = candidate.objectiveValue;
                    bestMoveCandidate = candidate;
                    bestNeighborhood = neighborhoodList.get(i);
                }
            }

            currentSolution = bestMoveCandidate;

            if (currentSolution.objectiveValue > bestSolution.objectiveValue) {
                bestSolution = new ALNSSolution(currentSolution);
                noImprovementIterations = 0;

                weightedNeighborhoods.merge(bestNeighborhood, 5.0, Double::sum);

                if (lastNeighborhood != null) {
                    int neighborhoodId = bestNeighborhood.id;
                    HashMap<Neighborhood, Integer> neighFactor = bondingFactors.get(neighborhoodId);
                    neighFactor.put(bestNeighborhood, Math.max(neighFactor.getOrDefault(bestNeighborhood, 0) + 1, 10));
                }

                lastNeighborhood = bestNeighborhood;
            } else if (currentSolution.objectiveValue > tempSolution.objectiveValue) {
                tempSolution = new ALNSSolution(currentSolution);

                weightedNeighborhoods.merge(bestNeighborhood, 2.0, Double::sum);

                lastNeighborhood = bestNeighborhood;
            } else {
                if (noImprovementIterations == maxNoImprovementIterations * 3) {
                    switch (rng.nextInt(4)) {
                        case 0:
                            currentSolution = new ALNSSolution(bestSolution);
                            break;

                        case 1:
                            currentSolution = getRandomSolution();
                            break;

                        case 2:
                            currentSolution = getOppositeSolution(bestSolution);
                            break;
                        case 3:
                            currentSolution = getHalfSolution(bestSolution);
                            break;
                    }
                    noImprovementIterations = 0;
                    lastNeighborhood = null;
                } else {
                    tempSolution = new ALNSSolution(currentSolution);
                    lastNeighborhood = bestNeighborhood;
                }   
            }

            currentPercentage = Math.min(maxPercentage,
                    minPercentage + (noImprovementIterations / (maxNoImprovementIterations / (maxPercentage - minPercentage))));

            evaporateWeights(weightedNeighborhoods);
        }

        executor.shutdown();

        return bestSolution;
    }

    private ALNSSolution convertToChallengeSolution(GRBVar[] orderVars, GRBVar[] aisleVars) throws GRBException {
        List<Integer> selectedOrders = new ArrayList<>();
        List<Integer> selectedAisles = new ArrayList<>();

        for (int i = 0; i < orderVars.length; i++) {
            if (orderVars[i].get(GRB.DoubleAttr.X) > 0.5) {
                selectedOrders.add(i);
            }
        }

        for (int i = 0; i < aisleVars.length; i++) {
            if (aisleVars[i].get(GRB.DoubleAttr.X) > 0.5) {
                selectedAisles.add(i);
            }
        }

        return new ALNSSolution(selectedOrders, selectedAisles);
    }

    public void evaporateWeights(HashMap<Neighborhood, Double> neighborhoods) {
        for (Map.Entry<Neighborhood, Double> entry : neighborhoods.entrySet()) {
            neighborhoods.put(entry.getKey(), Math.max(1.0, entry.getValue() * 0.8));
        }
    }

    public Neighborhood selectNeighborhood(
        HashMap<Neighborhood, Double> neighborhoods,
        Neighborhood lastNeighborhood) {

        Map<Neighborhood, Double> effectiveWeights = new HashMap<>();
        double totalWeight = 0.0;

        for (Map.Entry<Neighborhood, Double> entry : neighborhoods.entrySet()) {
            double weight = entry.getValue();
            if (lastNeighborhood != null) {
                weight += bondingFactors.get(entry.getKey().id).getOrDefault(lastNeighborhood, 0);
            }
            if (entry.getKey().equals(lastNeighborhood)) {
                weight *= 0.2;
            }

            effectiveWeights.put(entry.getKey(), weight);
            totalWeight += weight;
        }

        double r = Math.random() * totalWeight;
        double cumulative = 0.0;

        for (Map.Entry<Neighborhood, Double> entry : effectiveWeights.entrySet()) {
            cumulative += entry.getValue();
            if (r <= cumulative) return entry.getKey();
        }

        return neighborhoods.keySet().iterator().next();
    }

    private ALNSSolution getInitialViableSolution() {
        List<Integer> selectedOrders = new ArrayList<>();
        List<Integer> selectedAisles = new ArrayList<>();
        int totalItems = 0;
        HashMap<Integer, Integer> aisleCapacities = new HashMap<>();

        for (int i = 0; i < aisles.size(); i++) {
            selectedAisles.add(i);
            for (Map.Entry<Integer, Integer> entry : aisles.get(i).entrySet()) {
                aisleCapacities.put(entry.getKey(), entry.getValue() + aisleCapacities.getOrDefault(entry.getKey(), 0));
            }
        }

        for (int i = 0; i < orders.size(); i++) {
            int orderTotalItems = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
            if (totalItems + orderTotalItems > waveSizeUB) {
                continue;
            }

            boolean canFulfill = true;
            for (Map.Entry<Integer, Integer> entry : orders.get(i).entrySet()) {
                int itemIndex = entry.getKey();
                int itemQuantity = entry.getValue();
                if (aisleCapacities.getOrDefault(itemIndex, 0) < itemQuantity) {
                    canFulfill = false;
                    break;
                }
            }

            if (canFulfill) {
                selectedOrders.add(i);
                totalItems += orderTotalItems;

                for (Map.Entry<Integer, Integer> entry : orders.get(i).entrySet()) {
                    int itemIndex = entry.getKey();
                    int itemQuantity = entry.getValue();
                    aisleCapacities.put(itemIndex, aisleCapacities.get(itemIndex) - itemQuantity);
                }

                if (totalItems >= waveSizeLB) {
                    break;
                }
            }
        }

        ALNSSolution initialSolution = new ALNSSolution(selectedOrders, selectedAisles);
        initialSolution.objectiveValue = computeObjectiveFunction(initialSolution);
        initialSolution.calcItemsLeftInAisles();

        return initialSolution;
    }

    private ALNSSolution getRandomSolution() {
        List<Integer> selectedOrders = new ArrayList<>();
        List<Integer> selectedAisles = new ArrayList<>();

        // Randomly select orders
        for (int i = 0; i < orders.size(); i++) {
            if (Math.random() < 0.5) {  
                selectedOrders.add(i);
            }
        }

        // Randomly select aisles
        for (int i = 0; i < aisles.size(); i++) {
            if (Math.random() < 0.5) { 
                selectedAisles.add(i);
            }
        }

        ALNSSolution randomSolution = new ALNSSolution(selectedOrders, selectedAisles);
        
        boolean feasible = isSolutionFeasible(randomSolution);
        randomSolution.objectiveValue = feasible ? computeObjectiveFunction(randomSolution) : 0.0;
        randomSolution.calcItemsLeftInAisles();

        return randomSolution;
    }

    private ALNSSolution getOppositeSolution(ALNSSolution solution) {
        List<Integer> selectedOrders = new ArrayList<>();
        List<Integer> selectedAisles = new ArrayList<>();

        Set<Integer> orderSet = new HashSet<>(solution.selectedOrders);
        Set<Integer> aisleSet = new HashSet<>(solution.selectedAisles);

        for (int i = 0; i < orders.size(); i++) {
            if (!orderSet.contains(i)) {
                selectedOrders.add(i);
            }
        }

        for (int i = 0; i < aisles.size(); i++) {
            if (!aisleSet.contains(i)) {
                selectedAisles.add(i);
            }
        }

        ALNSSolution oppositeSolution = new ALNSSolution(selectedOrders, selectedAisles);

        boolean feasible = isSolutionFeasible(oppositeSolution);
        oppositeSolution.objectiveValue = feasible ? computeObjectiveFunction(oppositeSolution) : 0.0;
        oppositeSolution.calcItemsLeftInAisles();

        return oppositeSolution;
    }

    private ALNSSolution getHalfSolution(ALNSSolution solution) {
        List<Integer> selectedOrders = new ArrayList<>(solution.selectedOrders);
        List<Integer> selectedAisles = new ArrayList<>(solution.selectedAisles);

        Collections.shuffle(selectedOrders);
        Collections.shuffle(selectedAisles);

        selectedOrders = selectedOrders.subList(0, selectedOrders.size() / 2);
        selectedAisles = selectedAisles.subList(0, selectedAisles.size() / 2);

        ALNSSolution halfSolution = new ALNSSolution(selectedOrders, selectedAisles);

        boolean feasible = isSolutionFeasible(halfSolution);
        halfSolution.objectiveValue = feasible ? computeObjectiveFunction(halfSolution) : 0.0;
        halfSolution.calcItemsLeftInAisles();

        return halfSolution;
    }

    class Neighborhood {
        public int id;

        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {

            if (!isSolutionFeasible(currentSolution)) {
                currentSolution.objectiveValue = 0.0;
                return;
            }

            double quality = computeObjectiveFunction(currentSolution);
            currentSolution.objectiveValue = quality;
        }
    }

    class RemoveRandomAisles extends Neighborhood {
        public int id = 0;

        @Override
        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {
            if (currentSolution.selectedAisles.size() <= 1) {
                super.move(currentSolution, percentage, randomFactor);
                return;
            }

            Collections.shuffle(currentSolution.selectedAisles, ThreadLocalRandom.current());

            int nToRemove = (int) Math.ceil(currentSolution.selectedAisles.size() * (percentage / 100.0));
            nToRemove = Math.max(1, nToRemove);

            for (int i = 0; i < nToRemove; i++) {
                int aisle = currentSolution.selectedAisles.remove(0);

                for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                    currentSolution.itensLeftInAisles.merge(entry.getKey(), -entry.getValue(), Integer::sum);
                }
            }

            super.move(currentSolution, percentage, randomFactor);
        }
    }

    class RemoveWorstAisles extends Neighborhood {
        public int id = 1;

        @Override
        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {
            if (currentSolution.selectedAisles.size() <= 1) {
                super.move(currentSolution, percentage, randomFactor);
                return;
            }

            Set<Integer> selectedSet = new HashSet<>(currentSolution.selectedAisles);
            List<Integer> candidateAisles = bestAislesByItemNumber.stream()
                .filter(a -> selectedSet.contains(a))
                .collect(Collectors.toList());
            Collections.reverse(candidateAisles);

            int nToRemove = (int) Math.ceil(currentSolution.selectedAisles.size() * (percentage / 100.0));
            nToRemove = Math.max(1, nToRemove);

            for (int i = 0; nToRemove > 0 && i < candidateAisles.size(); i++) {
                int aisle = candidateAisles.get(i);

                if (ThreadLocalRandom.current().nextDouble() > randomFactor) continue;

                currentSolution.selectedAisles.remove(Integer.valueOf(aisle));
                nToRemove--;

                for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                    currentSolution.itensLeftInAisles.merge(entry.getKey(), -entry.getValue(), Integer::sum);
                }
            }

            super.move(currentSolution, percentage, randomFactor);
        }
    }

    class RemoveRandomOrders extends Neighborhood {
        public int id = 2;

        @Override
        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {
            if (currentSolution.selectedOrders.size() < 1) {
                super.move(currentSolution, percentage, randomFactor);
                return;
            }

            Collections.shuffle(currentSolution.selectedOrders, ThreadLocalRandom.current());

            int nToRemove = (int) Math.ceil(currentSolution.selectedOrders.size() * (percentage / 100.0));
            nToRemove = Math.max(1, nToRemove);

            for (int i = 0; i < nToRemove; i++) {
                int order = currentSolution.selectedOrders.remove(0);

                for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                    currentSolution.itensLeftInAisles.merge(entry.getKey(), entry.getValue(), Integer::sum);
                    currentSolution.totalItemsPicked -= entry.getValue();
                }
            }

            super.move(currentSolution, percentage, randomFactor);
        }
    }

    class RemoveWorstOrders extends Neighborhood {
        public int id = 3;

        @Override
        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {
            if (currentSolution.selectedOrders.size() < 1) {
                super.move(currentSolution, percentage, randomFactor);
                return;
            }

            Set<Integer> selectedSet = new HashSet<>(currentSolution.selectedOrders);
            List<Integer> candidateOrders = bestOrdersByItemNumber.stream()
                .filter(o -> selectedSet.contains(o))
                .collect(Collectors.toList());
            Collections.reverse(candidateOrders);

            int nToRemove = (int) Math.ceil(currentSolution.selectedOrders.size() * (percentage / 100.0));
            nToRemove = Math.max(1, nToRemove);

            for (int i = 0; nToRemove > 0 && i < candidateOrders.size(); i++) {
                int order = candidateOrders.get(i);

                if (ThreadLocalRandom.current().nextDouble() > randomFactor) continue;

                currentSolution.selectedOrders.remove(Integer.valueOf(order));
                nToRemove--;

                for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                    currentSolution.itensLeftInAisles.merge(entry.getKey(), entry.getValue(), Integer::sum);
                    currentSolution.totalItemsPicked -= entry.getValue();
                }
            }

            super.move(currentSolution, percentage, randomFactor);
        }
    }

    class AddBestOrders extends Neighborhood {
        public int id = 4;

        @Override
        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {
            Set<Integer> selectedSet = new HashSet<>(currentSolution.selectedOrders);
            List<Integer> candidateOrders = bestOrdersByItemNumber.stream()
                .filter(o -> !selectedSet.contains(o))
                .collect(Collectors.toList());

            if (candidateOrders.isEmpty()) {
                super.move(currentSolution, percentage, randomFactor);
                return;
            }

            for (int order : candidateOrders) {
                Map<Integer, Integer> orderItems = orders.get(order);
                int orderTotalItems = orderItems.values().stream().mapToInt(Integer::intValue).sum();

                if (currentSolution.totalItemsPicked + orderTotalItems > waveSizeUB) {
                    continue;
                }

                boolean canFulfill = true;
                for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                    int itemIndex = entry.getKey();
                    int itemQuantity = entry.getValue();
                    if (currentSolution.itensLeftInAisles.getOrDefault(itemIndex, 0) < itemQuantity) {
                        canFulfill = false;
                        break;
                    }
                }

                if (canFulfill && ThreadLocalRandom.current().nextDouble() < randomFactor) {
                    currentSolution.selectedOrders.add(order);
                    currentSolution.totalItemsPicked += orderTotalItems;
                    for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                        int itemIndex = entry.getKey();
                        int itemQuantity = entry.getValue();
                        currentSolution.itensLeftInAisles.merge(itemIndex, -itemQuantity, Integer::sum);
                    }
                }
            }

            super.move(currentSolution, percentage, randomFactor);
        }
    }

    class AddBestAisles extends Neighborhood {
        public int id = 5;

        @Override
        public void move(ALNSSolution currentSolution, double percentage, double randomFactor) {
            Set<Integer> selectedSet = new HashSet<>(currentSolution.selectedAisles);
            List<Integer> candidateAisles = bestAislesByItemNumber.stream()
                .filter(a -> !selectedSet.contains(a))
                .collect(Collectors.toList());

            if (candidateAisles.isEmpty()) {
                super.move(currentSolution, percentage, randomFactor);
                return;
            }

            boolean valid = false;

            for (int aisle : candidateAisles) {
                if (ThreadLocalRandom.current().nextDouble() < randomFactor) {
                    currentSolution.selectedAisles.add(aisle);
                    for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                        currentSolution.itensLeftInAisles.merge(entry.getKey(), entry.getValue(), Integer::sum);
                    }
                }

                if (valid && ThreadLocalRandom.current().nextDouble() < randomFactor) {
                    break;
                } else if (currentSolution.itensLeftInAisles.values().stream().allMatch(v -> v >= 0)) {
                    valid = true;
                }
            }

            super.move(currentSolution, percentage, randomFactor);
        }
    }

    private GRBEnv initializeGrbEnv(StopWatch stopWatch, int nThreads) {
        try {
            GRBEnv env = new GRBEnv(true);
            env.set("logFile", "mip1.log");
            env.set(GRB.DoubleParam.TimeLimit, 60.0);
            env.set(GRB.IntParam.LogToConsole, 0);
            env.set(GRB.IntParam.Threads, nThreads);
            env.start();
            return env;
        } catch (GRBException e) {
            System.err.println("Error initializing Gurobi environment.");
            e.printStackTrace();
            return null;
        }
    }

    /*
     * Get the remaining time in seconds
     */
    protected long getRemainingTime(StopWatch stopWatch) {
        return Math.max(
                TimeUnit.SECONDS.convert(MAX_RUNTIME - stopWatch.getTime(TimeUnit.MILLISECONDS), TimeUnit.MILLISECONDS),
                0);
    }

    protected boolean isSolutionFeasible(ALNSSolution challengeSolution) {
        List<Integer> selectedOrders = challengeSolution.selectedOrders;
        List<Integer> visitedAisles = challengeSolution.selectedAisles;
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

    protected double computeObjectiveFunction(ALNSSolution challengeSolution) {
        List<Integer> selectedOrders = challengeSolution.selectedOrders;
        List<Integer> visitedAisles = challengeSolution.selectedAisles;
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