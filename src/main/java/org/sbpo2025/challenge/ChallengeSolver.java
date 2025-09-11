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

    protected final int nThreads = 4;


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

    private interface LocalSearchStrategy {
        void apply(ILSSolution sol, Random rand);
    }

    private final List<LocalSearchStrategy> strategies = Arrays.asList(
        (sol, rand) -> { tryAddBestOrders(sol); },   // comb 1
        (sol, rand) -> { tryRemoveWorstAisles(sol); },   // comb 2
        (sol, rand) -> { tryAddRandomOrders(sol, rand); }, // comb 3
        (sol, rand) -> { tryRemoveRandomAisles(sol, rand); }   // comb 4
    );


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
        while (stopWatch.getTime(TimeUnit.SECONDS) < 60) { // critério de parada
            // --- Busca Local
            current = localSearchParallel(current, nThreads, rnd);

            // --- Atualiza melhor
            if (current.objectiveValue > best.objectiveValue) {
                alpha = 0.2;
                best = new ILSSolution(current);
            } else {
                if(alpha<0.40)
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

    private ILSSolution localSearchParallel(ILSSolution sol, int nThreads, Random rand) {
        ExecutorService executor = Executors.newFixedThreadPool(nThreads);
        List<Callable<ILSSolution>> tasks = new ArrayList<>();

        // pega até nThreads combinações
        for (int i = 0; i < nThreads && i < strategies.size(); i++) {
            int idx = i;
            tasks.add(() -> {
                ILSSolution copy = new ILSSolution(sol); // cada thread trabalha numa cópia
                strategies.get(idx).apply(copy, new Random(rand.nextLong())); // usa seed diferente
                copy.objectiveValue = computeObjectiveFunction(copy);
                return copy;
            });
        }

        try {
            List<Future<ILSSolution>> results = executor.invokeAll(tasks);
            executor.shutdown();

            // pega o melhor resultado
            ILSSolution bestLocal = sol;
            for (Future<ILSSolution> f : results) {
                ILSSolution candidate = f.get();
                if (candidate.objectiveValue > bestLocal.objectiveValue) {
                    bestLocal = candidate;
                }
            }
            return bestLocal;
        } catch (Exception e) {
            e.printStackTrace();
            return sol;
        }
    }

    private boolean tryAddBestOrder(ILSSolution sol) {
        Set<Integer> selectedOrders = new HashSet<>(sol.selectedOrders);

        for (int order : bestOrdersByItemNumber) {
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

            if (canFulfill) {
                sol.selectedOrders.add(order);
                sol.totalItemsPicked += orderTotalItems;

                for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                    sol.itensLeftInAisles.merge(entry.getKey(), -entry.getValue(), Integer::sum);
                }

                sol.objectiveValue = computeObjectiveFunction(sol);
                return true;
            }
        }

        return false;
    }

    private void tryAddBestOrders(ILSSolution sol) {
        Set<Integer> selectedOrders = new HashSet<>(sol.selectedOrders);
        for (int order : bestOrdersByItemNumber) {
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

            if (canFulfill) {
                sol.selectedOrders.add(order);
                sol.totalItemsPicked += orderTotalItems;

                for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                    sol.itensLeftInAisles.merge(entry.getKey(), -entry.getValue(), Integer::sum);
                }
            } else break;
        }
    }

    private void tryAddRandomOrders(ILSSolution sol, Random rnd) {
        Set<Integer> selectedOrders = new HashSet<>(sol.selectedOrders);

        // pega todos os pedidos ainda não selecionados
        List<Integer> candidateOrders = IntStream.range(0, orders.size())
                .filter(o -> !selectedOrders.contains(o))
                .boxed()
                .collect(Collectors.toList());

        // embaralha para iterar em ordem aleatória
        Collections.shuffle(candidateOrders, rnd);

        for (int order : candidateOrders) {
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

            if (canFulfill) {
                sol.selectedOrders.add(order);
                sol.totalItemsPicked += orderTotalItems;

                for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                    sol.itensLeftInAisles.merge(entry.getKey(), -entry.getValue(), Integer::sum);
                }
            } else break;
        }
    }


    private boolean tryRemoveWorstAisle(ILSSolution sol) {
        if (sol.selectedAisles.size() <= 1) return false;

        Set<Integer> selectedSet = new HashSet<>(sol.selectedAisles);

        // pega apenas os corredores já selecionados, ordenados do pior para o melhor
        List<Integer> candidateAisles = bestAislesByItemNumber.stream()
                .filter(selectedSet::contains)
                .collect(Collectors.toList());
        Collections.reverse(candidateAisles); // corredores com menos itens primeiro

        for (int aisle : candidateAisles) {
            boolean canRemove = true;

            // simula a remoção do corredor
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                int item = entry.getKey();
                int qty = entry.getValue();

                // se já está no limite de suprimento, não dá para remover
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

    private void tryRemoveWorstAisles(ILSSolution sol) {
        if (sol.selectedAisles.size() <= 1) return;

        Set<Integer> selectedSet = new HashSet<>(sol.selectedAisles);

        // pega apenas os corredores já selecionados, ordenados do pior para o melhor
        List<Integer> candidateAisles = bestAislesByItemNumber.stream()
                .filter(selectedSet::contains)
                .collect(Collectors.toList());
        Collections.reverse(candidateAisles); // corredores com menos itens primeiro

        for (int aisle : candidateAisles) {
            boolean canRemove = true;

            // simula a remoção do corredor
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                int item = entry.getKey();
                int qty = entry.getValue();

                // se já está no limite de suprimento, não dá para remover
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
            } else break;
        }
    }

    private void tryRemoveRandomAisles(ILSSolution sol, Random rnd) {
        if (sol.selectedAisles.size() <= 1) return;

        List<Integer> candidateAisles = new ArrayList<>(sol.selectedAisles);

        // embaralha a lista para iterar em ordem aleatória
        Collections.shuffle(candidateAisles, rnd);

        for (int aisle : candidateAisles) {
            boolean canRemove = true;

            // simula a remoção do corredor
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                int item = entry.getKey();
                int qty = entry.getValue();

                // se já está no limite de suprimento, não dá para remover
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
            } else break;
        }
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
