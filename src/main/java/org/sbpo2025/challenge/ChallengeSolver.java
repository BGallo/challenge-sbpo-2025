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
    private Map<Integer, List<Integer>> itemInAisles = new HashMap<>();
    private Map<Integer, List<Integer>> itemInOrders = new HashMap<>();

    class EnhancedSolution {
        ChallengeSolution base;       
        Map<Integer, Integer> itemsLeft;

        public EnhancedSolution(ChallengeSolution base, Map<Integer, Integer> itemsLeft) {
            this.base = base;
            this.itemsLeft = itemsLeft;
        }

        public EnhancedSolution clone() {
            return new EnhancedSolution(
                new ChallengeSolution(this.base.orders(), this.base.orders()),
                new HashMap<>(itemsLeft)
            );
        }
    }


    public ChallengeSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB, int waveSizeUB) {
        this.orders = orders;
        this.aisles = aisles;
        this.nItems = nItems;
        this.waveSizeLB = waveSizeLB;
        this.waveSizeUB = waveSizeUB;
    }

    public ChallengeSolution solve(StopWatch stopWatch) {

        buildItemStructures();

        return new ChallengeSolution(new HashSet<>(), new HashSet<>());
    }

    private void buildItemStructures() {
        for (int aisleIndex = 0; aisleIndex < aisles.size(); aisleIndex++) {
            for (int item : aisles.get(aisleIndex).keySet()) {
                itemInAisles.computeIfAbsent(item, k -> new ArrayList<>()).add(aisleIndex);
            }
        }

        for (int orderIndex = 0; orderIndex < orders.size(); orderIndex++) {
            for (int item : orders.get(orderIndex).keySet()) {
                itemInOrders.computeIfAbsent(item, k -> new ArrayList<>()).add(orderIndex);
            }
        }
    }

    private Map<Integer, Integer> buildItemsLeft(List<Integer> selectedAisles, List<Integer> selectedOrders,
                                            List<Map<Integer,Integer>> aisles, List<Map<Integer,Integer>> orders) {
        Map<Integer, Integer> supply = new HashMap<>();

        // soma itens dos corredores escolhidos
        for (int aisle : selectedAisles) {
            for (Map.Entry<Integer, Integer> entry : aisles.get(aisle).entrySet()) {
                supply.put(entry.getKey(), supply.getOrDefault(entry.getKey(), 0) + entry.getValue());
            }
        }

        // desconta itens usados nos pedidos
        for (int order : selectedOrders) {
            for (Map.Entry<Integer, Integer> entry : orders.get(order).entrySet()) {
                int item = entry.getKey();
                int demand = entry.getValue();
                supply.put(item, supply.getOrDefault(item, 0) - demand);
            }
        }

        // no final, valores positivos = itens sobrando
        return supply;
    }

    public EnhancedSolution ILS(EnhancedSolution initial, int maxIterations, double alpha) {
        EnhancedSolution best = initial.clone();
        EnhancedSolution current = initial.clone();

        for (int it = 0; it < maxIterations; it++) {
            // 🔹 Busca local: tenta melhorar retirando corredores e depois adicionando pedidos
            current = localSearch(current, alpha);

            if (computeObjectiveFunction(current.base) > computeObjectiveFunction(best.base)) {
                best = current.clone();
            }

            // 🔹 Perturbação: remove alguns pedidos e tenta recolocar outros
            current = perturbation(best.clone(), alpha);
        }

        return best;
    }

    private EnhancedSolution localSearch(EnhancedSolution sol, double alpha) {
        boolean improved = true;
        while (improved) {
            improved = false;

            // 1. tenta remover corredores que só têm sobra (usando itemsLeft)
            for (int aisle : new ArrayList<>(sol.base.aisles())) {
                if (canRemoveAisle(sol, aisle)) {
                    removeAisle(sol, aisle);
                    improved = true;
                    break;
                }
            }

            // 2. tenta adicionar pedidos se houver sobra suficiente em itemsLeft
            for (int order : getCandidateOrders()) {
                if (!sol.base.orders().contains(order) && canAddOrder(sol, order)) {
                    addOrder(sol, order);
                    improved = true;
                    break;
                }
            }
        }
        return sol;
    }

    private EnhancedSolution perturbation(EnhancedSolution sol, double alpha) {
        Random rand = new Random();

        // remove aleatoriamente k pedidos
        int k = Math.max(1, sol.base.orders().size() / 4);
        List<Integer> ordersList = new ArrayList<>(sol.base.orders());
        Collections.shuffle(ordersList);

        for (int i = 0; i < k; i++) {
            removeOrder(sol, ordersList.get(i));
        }

        // tenta recolocar novos pedidos usando itemsLeft
        for (int order : getCandidateOrders()) {
            if (!sol.base.orders().contains(order) && canAddOrder(sol, order)) {
                addOrder(sol, order);
            }
        }

        return sol;
    }


    // Retorna todos os pedidos que podem ser considerados (excluindo os já escolhidos)
    private List<Integer> getCandidateOrders() {
        List<Integer> candidates = new ArrayList<>();
        for (int i = 0; i < orders.size(); i++) {
            candidates.add(i);
        }
        return candidates;
    }

    // Verifica se é possível adicionar o pedido sem faltar itens
    private boolean canAddOrder(EnhancedSolution sol, int order) {
        Map<Integer, Integer> demand = orders.get(order);
        for (Map.Entry<Integer, Integer> entry : demand.entrySet()) {
            int item = entry.getKey();
            int qty = entry.getValue();
            // precisa ter sobra suficiente
            if (sol.itemsLeft.getOrDefault(item, 0) < qty) {
                return false;
            }
        }
        return true;
    }

    // Adiciona o pedido e atualiza itemsLeft
    private void addOrder(EnhancedSolution sol, int order) {
        sol.base.orders().add(order);
        Map<Integer, Integer> demand = orders.get(order);
        for (Map.Entry<Integer, Integer> entry : demand.entrySet()) {
            int item = entry.getKey();
            int qty = entry.getValue();
            sol.itemsLeft.put(item, sol.itemsLeft.getOrDefault(item, 0) - qty);
        }
    }

    // Verifica se o corredor pode ser removido (ou seja, todos os itens que ele fornece estão sobrando)
    private boolean canRemoveAisle(EnhancedSolution sol, int aisle) {
        Map<Integer, Integer> supply = aisles.get(aisle);
        for (Map.Entry<Integer, Integer> entry : supply.entrySet()) {
            int item = entry.getKey();
            int qty = entry.getValue();
            // só pode remover se tiver sobra suficiente desse item em itemsLeft
            if (sol.itemsLeft.getOrDefault(item, 0) < qty) {
                return false;
            }
        }
        return true;
    }

    // Remove o corredor e atualiza itemsLeft
    private void removeAisle(EnhancedSolution sol, int aisle) {
        sol.base.aisles().remove((Integer) aisle);
        Map<Integer, Integer> supply = aisles.get(aisle);
        for (Map.Entry<Integer, Integer> entry : supply.entrySet()) {
            int item = entry.getKey();
            int qty = entry.getValue();
            sol.itemsLeft.put(item, sol.itemsLeft.getOrDefault(item, 0) - qty);
        }
    }


    // Remove um pedido e devolve seus itens para itemsLeft
    private void removeOrder(EnhancedSolution sol, int order) {
        sol.base.orders().remove((Integer) order);
        Map<Integer, Integer> demand = orders.get(order);
        for (Map.Entry<Integer, Integer> entry : demand.entrySet()) {
            int item = entry.getKey();
            int qty = entry.getValue();
            sol.itemsLeft.put(item, sol.itemsLeft.getOrDefault(item, 0) + qty);
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

    protected double computeObjectiveFunction(ChallengeSolution challengeSolution) {
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();
        if (selectedOrders == null || visitedAisles == null || selectedOrders.isEmpty() || visitedAisles.isEmpty()) {
            return 0.0;
        }

        int penalty = 0;
        if (!isSolutionFeasible(challengeSolution)) {
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
        return (double) totalUnitsPicked / numVisitedAisles + penalty;
    }

    protected int applyPenalty(ChallengeSolution challengeSolution) {
        int penalty = 0;
        Set<Integer> selectedOrders = challengeSolution.orders();
        Set<Integer> visitedAisles = challengeSolution.aisles();

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
