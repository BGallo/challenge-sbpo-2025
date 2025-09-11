package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;
import org.sbpo2025.challenge.alns.ALNSSolution;

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

    protected List<Integer> bestOrdersByItemNumber = new ArrayList<>();
    protected List<Integer> bestAislesByItemNumber = new ArrayList<>();

    protected final int nThreads = 8;


    public ChallengeSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB, int waveSizeUB) {
        this.orders = orders;
        this.aisles = aisles;
        this.nItems = nItems;
        this.waveSizeLB = waveSizeLB;
        this.waveSizeUB = waveSizeUB;



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

    class ILSSolution {
        List<Integer> selectedAisles = new ArrayList<>();
        List<Integer> selectedOrders = new ArrayList<>();
        Map<Integer, Integer> itensLeftInAisles = new HashMap<>();
        double objectiveValue;
        int totalItemsPicked;

        public ILSSolution() {
            this.objectiveValue = 0;
        }

        public ILSSolution(List<Integer> selectedOrders, List<Integer> selectedAisles) {
            this.selectedOrders = selectedOrders;
            this.selectedAisles = selectedAisles;
        }

        public ILSSolution(ILSSolution other) {
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

    public ChallengeSolution solve(StopWatch stopWatch) {


        ILSSolution best = getInitialViableSolution();
        best.calcItemsLeftInAisles();
        best.objectiveValue = computeObjectiveFunction(best);

        System.out.println("Solução Inicial: " + computeObjectiveFunction(best));
        System.out.println("feasible:" + isSolutionFeasible(best));

        ILSSolution current = new ILSSolution(best);
        Random rnd = ThreadLocalRandom.current();
        double alpha = 0.2;

        int iter = 0;
        while (stopWatch.getTime(TimeUnit.SECONDS) < 30) { // critério de parada
            // --- Busca Local
            localSearch(current);

            // --- Atualiza melhor
            if (current.objectiveValue > best.objectiveValue) {
                alpha = 0.2;
                best = new ILSSolution(current);
            } else {
                alpha+=0.01;
            }

            // --- Perturbação
            ILSSolution perturbed = perturb(current, rnd, alpha);

            // --- Aceitação (simples: aceita sempre, ou Simulated Annealing style)
            current = perturbed;

            iter++;
        }

        System.out.println("Iterações: " + iter);
        System.out.println("Solução: " + computeObjectiveFunction(best));
        System.out.println("feasible:" + isSolutionFeasible(best));
        return new ChallengeSolution(
            new HashSet<>(best.selectedOrders),
            new HashSet<>(best.selectedAisles)
        );
    }

    private ILSSolution getInitialViableSolution() {
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

        ILSSolution initialSolution = new ILSSolution(selectedOrders, selectedAisles);
        initialSolution.objectiveValue = computeObjectiveFunction(initialSolution);
        initialSolution.calcItemsLeftInAisles();

        return initialSolution;
    }

    private void localSearch(ILSSolution sol) {
        boolean improved = true;
        while (improved) {
            improved = false;

            // Tenta adicionar ordens boas
            if (tryAddBestOrder(sol)) {
                improved = true;
                continue;
            }

            // Tenta remover corredores ruins
            if (tryRemoveWorstAisle(sol)) {
                improved = true;
            }
        }
    }

    private boolean tryAddBestOrder(ILSSolution sol) {
        Set<Integer> selectedOrders = new HashSet<>(sol.selectedOrders);

        int bestGain = 0;
        Integer bestOrder = null;

        for (int order = 0; order < orders.size(); order++) {
            if (selectedOrders.contains(order)) continue;

            Map<Integer, Integer> orderItems = orders.get(order);
            int orderTotalItems = orderItems.values().stream().mapToInt(Integer::intValue).sum();

            if (sol.totalItemsPicked + orderTotalItems > waveSizeUB) continue;

            boolean canFulfill = true;
            for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                int item = entry.getKey();
                int qty = entry.getValue();
                if (sol.itensLeftInAisles.getOrDefault(item, 0) < qty) {
                    canFulfill = false;
                    break;
                }
            }

            if (canFulfill && orderTotalItems > bestGain) {
                bestGain = orderTotalItems;
                bestOrder = order;
            }
        }

        if (bestOrder != null) {
            sol.selectedOrders.add(bestOrder);
            int orderTotalItems = orders.get(bestOrder).values().stream().mapToInt(Integer::intValue).sum();
            sol.totalItemsPicked += orderTotalItems;

            for (Map.Entry<Integer, Integer> entry : orders.get(bestOrder).entrySet()) {
                sol.itensLeftInAisles.merge(entry.getKey(), -entry.getValue(), Integer::sum);
            }

            sol.objectiveValue = computeObjectiveFunction(sol);
            return true;
        }

        return false;
    }

    private boolean tryRemoveWorstAisle(ILSSolution sol) {
        if (sol.selectedAisles.size() <= 1) return false;

        for (int aisle : new ArrayList<>(sol.selectedAisles)) {
            boolean canRemove = true;

            // simula a remoção do corredor
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                int item = entry.getKey();
                int qty = entry.getValue();

                // Se já está no limite de suprimento, não dá para remover
                if (sol.itensLeftInAisles.getOrDefault(item, 0) + qty < 0) {
                    canRemove = false;
                    break;
                }
            }

            if (canRemove) {
                sol.selectedAisles.remove(Integer.valueOf(aisle));
                for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                    sol.itensLeftInAisles.merge(entry.getKey(), -entry.getValue(), Integer::sum);
                }
                sol.objectiveValue = computeObjectiveFunction(sol);
                return true;
            }
        }

        return false;
    }



    private ILSSolution perturb(ILSSolution sol, Random rnd, double alpha) {
        ILSSolution newSol = new ILSSolution(sol);

        // ---- REMOVER ORDENS ----
        int nRemove = Math.max(1, (int) Math.ceil(alpha * newSol.selectedOrders.size()));
        Collections.shuffle(newSol.selectedOrders, rnd);
        for (int i = 0; i < nRemove && !newSol.selectedOrders.isEmpty(); i++) {
            int order = newSol.selectedOrders.remove(0);
            for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                newSol.itensLeftInAisles.merge(entry.getKey(), entry.getValue(), Integer::sum);
                newSol.totalItemsPicked -= entry.getValue();
            }
        }

        // ---- ADICIONAR CORREDORES ----
        List<Integer> notSelected = IntStream.range(0, aisles.size())
                .filter(a -> !newSol.selectedAisles.contains(a))
                .boxed().collect(Collectors.toList());

        if (!notSelected.isEmpty()) {
            int nAdd = Math.max(1, (int) Math.ceil(alpha * notSelected.size()));
            Collections.shuffle(notSelected, rnd);

            for (int i = 0; i < nAdd && i < notSelected.size(); i++) {
                int aisle = notSelected.get(i);
                newSol.selectedAisles.add(aisle);
                for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                    newSol.itensLeftInAisles.merge(entry.getKey(), entry.getValue(), Integer::sum);
                }
            }
        }

        newSol.calcItemsLeftInAisles();
        newSol.objectiveValue = computeObjectiveFunction(newSol);
        return newSol;
    }




    /*
     * Get the remaining time in seconds
     */
    protected long getRemainingTime(StopWatch stopWatch) {
        return Math.max(
                TimeUnit.SECONDS.convert(MAX_RUNTIME - stopWatch.getTime(TimeUnit.MILLISECONDS), TimeUnit.MILLISECONDS),
                0);
    }

    protected boolean isSolutionFeasible(ILSSolution initialSolution) {
            List<Integer> selectedOrders = initialSolution.selectedOrders;
            List<Integer> visitedAisles = initialSolution.selectedAisles;
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

    protected double computeObjectiveFunction(ILSSolution initialSolution) {
            List<Integer> selectedOrders = initialSolution.selectedOrders;
            List<Integer> visitedAisles = initialSolution.selectedAisles;
            if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
                return 0.0;
            }
    
            int penalty = 0;
            if (!isSolutionFeasible(initialSolution)) {
                penalty -= applyPenalty(initialSolution);
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
        return (double) totalUnitsPicked / numVisitedAisles + penalty;
    }

    protected int applyPenalty(ILSSolution initialSolution) {
            int penalty = 0;
            List<Integer> selectedOrders = initialSolution.selectedOrders;
            List<Integer> visitedAisles = initialSolution.selectedAisles;

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
}
