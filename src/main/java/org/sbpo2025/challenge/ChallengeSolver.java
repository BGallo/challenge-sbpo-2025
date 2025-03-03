package org.sbpo2025.challenge;

import java.time.LocalTime;
/* import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException; */
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.apache.commons.lang3.time.StopWatch;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 540000; // milliseconds; 30 s
    protected List<Map<Integer, Integer>> orders;
    protected List<Map<Integer, Integer>> aisles;
    protected int nItems;
    protected int waveSizeLB;
    protected int waveSizeUB;

    public ChallengeSolver(
            List<Map<Integer, Integer>> orders, List<Map<Integer, Integer>> aisles, int nItems, int waveSizeLB,
            int waveSizeUB) {
        this.orders = orders;
        this.aisles = aisles;
        this.nItems = nItems;
        this.waveSizeLB = waveSizeLB;
        this.waveSizeUB = waveSizeUB;
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

    /* public static void writeNumberToFile(String number, String fileName) {
        File file = new File(fileName);
        File parentDir = file.getParentFile();
  
        if (parentDir != null && !parentDir.exists()) {
            parentDir.mkdirs();
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file, true))) {
            writer.write(" ");
            writer.write(number);
            System.out.println("Número adicionado ao arquivo com sucesso.");
        } catch (IOException e) {
            System.err.println("Ocorreu um erro ao escrever no arquivo.");
            e.printStackTrace();
        }
    } */

    public ChallengeSolution solve(StopWatch stopWatch) {
        double alpha = 0.50;

        long startTime = System.currentTimeMillis();

        RVNDSolution bestSolution = constructGreedyRandomizedSolution(alpha);
        double bestQuality = computeObjectiveFunction(decodeRVNDSolution(bestSolution));

        System.out.println("Initial Solution with quality: " + bestQuality);
        System.out.println("is Feaseble: " + isSolutionFeasible(bestSolution, true));

        int currentIteration = 0;
        /* int maxIterations = 1; */
        int maxIterations = Integer.MAX_VALUE;
        Shake shake = new Shake();
        int shakeTimer = 5;
        int greedyTimer = 7;
        while (System.currentTimeMillis() - startTime < MAX_RUNTIME && currentIteration < maxIterations) {
            currentIteration++;

            /* System.out.println("Iteration " + currentIteration); */

            RVNDSolution currentSolution;

            if (currentIteration % greedyTimer == 0) {
                currentSolution = constructGreedyRandomizedSolution(alpha);
            } else if (currentIteration % shakeTimer == 0) {
                currentSolution = shake.explore(bestSolution, bestQuality);
            } else {
                currentSolution = bestSolution;
            }

            double currentQuality = bestQuality;

            currentSolution = randomVariableNeighborhoodDescent(currentSolution, currentQuality);
            currentQuality = computeObjectiveFunction(decodeRVNDSolution(currentSolution));

            if (currentQuality > bestQuality) {
                bestQuality = currentQuality;
                bestSolution = currentSolution;
                
                /* System.out.println("New best solution found in Iteration " + currentIteration
                  + ": " + bestQuality); */
            }
        }

        double graspFitness = 0.0;

        if (bestSolution != null) {
            graspFitness = computeObjectiveFunction(decodeRVNDSolution(bestSolution));
            System.out.println("Best Score GRASP:" + graspFitness);
            System.out.println("É viável: " + isSolutionFeasible(bestSolution, true));
            /* System.out.println("Solution: " + bestSolution); */
        }

       /*  String formattedNumber = String.format("%.3f", graspFitness);
        writeNumberToFile(formattedNumber, "number_output.txt");  */  

        return new ChallengeSolution(new HashSet<>(bestSolution.orders()), new HashSet<>(bestSolution.aisles()));
    }

    private RVNDSolution constructGreedyRandomizedSolution(double alpha) {
        Random rand = new Random();

        // base responses
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();

        // the index of the candidates
        List<Integer> candidateOrders = new ArrayList<>();

        // variables that determine the number of items in the order
        Map<Integer, Integer> orderItemCount = new HashMap<>();

        for (int i = 0; i < orders.size(); i++) {
            int totalItems = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
            orderItemCount.put(i, totalItems);
            candidateOrders.add(i);
        }

        int totalSelectedItems = 0;

        while (!candidateOrders.isEmpty()) {
            List<Integer> RCL = buildRCL(candidateOrders, orderItemCount, alpha);

            if (RCL.isEmpty())
                break;
            /* System.out.println(RCL); */

            int selectedOrder = RCL.get(rand.nextInt(RCL.size()));
            int selectedOrderItems = orderItemCount.get(selectedOrder);

            /*
             * System.out.println("The order selected is " + selectedOrder + " with " +
             * selectedOrderItems + " items");
             */

            if (totalSelectedItems + selectedOrderItems > waveSizeUB) {
                candidateOrders.remove(Integer.valueOf(selectedOrder));
                /*
                 * System.out.
                 * println("The order was removed because it exceeded the wave size UB");
                 */
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

    private List<Integer> buildRCL(List<Integer> candidateOrders, Map<Integer, Integer> orderItemCount, double alpha) {
        int maxItems = candidateOrders.parallelStream().mapToInt(orderItemCount::get).max().orElse(0);
        int minItems = candidateOrders.parallelStream().mapToInt(orderItemCount::get).min().orElse(0);
        double threshold = maxItems - alpha * (maxItems - minItems);

        List<Integer> RCL = candidateOrders.parallelStream()
                .filter(order -> orderItemCount.get(order) >= threshold)
                .collect(Collectors.toList());

        return RCL;
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

                boolean aisleHasSomeItem = this.aisles.get(aisle).keySet().parallelStream()
                        .anyMatch(item -> currentOrder.containsKey(item));

                if (!aisleHasSomeItem)
                    continue;

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

    class Neighborhood {
        protected final int MAX_ITERATIONS = 50;

        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality) {
            return null;
        }

        boolean move(RVNDSolution solution, int iterator) {
            return false;
        }
    }

    class RemoveAisle extends Neighborhood {

        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality) {
            /* System.out.println("Exploring Remove Aisle Neighborhood"); */
            RVNDSolution bestSolution = initialSolution;

            /* System.out.println("Initial Solution: " + initialSolution); */

            bestSolution = IntStream.range(0, initialSolution.aisles().size()).limit(this.MAX_ITERATIONS).parallel().mapToObj((aisleIndex) -> {
                RVNDSolution currentSolution = new RVNDSolution(
                        new ArrayList<>(initialSolution.orders()),
                        new ArrayList<>(initialSolution.aisles()));

                if (!this.move(currentSolution, aisleIndex)) {
                    return null;
                }

                double currentQuality = computeObjectiveFunction(decodeRVNDSolution(currentSolution));

                return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
            }).filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey)
                    .orElse(bestSolution);

            /* System.out.println("Finished Exploring Remove Aisle Neighborhood " + bestSolution); */
            return bestSolution;
        }

        // A move configures a removal of an aisle and, from there, the removal of any
        // other aisle that can be removed
        @Override
        boolean move(RVNDSolution solution, int iterator) {
            /* System.out.println(iterator + ": Performing a move in Remove Aisle Neighborhood " + iterator); */
            if (iterator >= solution.aisles().size())
                return false;

            solution.aisles().remove(iterator);

            /* System.out.println(iterator + ": Solution after removing aisle " + iterator + ": " + solution + " is feasible: "
                    + isSolutionFeasible(solution, false)); */

            if (!isSolutionFeasible(solution, false))
                return false;

            int nextRemovableAisle = getNextRemovableAisle(solution);
            while (nextRemovableAisle != -1) {
                /* System.out.println(iterator + ": More aisles could be removed, removing aisle " + nextRemovableAisle); */
                solution.aisles().remove(nextRemovableAisle);
                nextRemovableAisle = getNextRemovableAisle(solution);
            }

            /* System.out.println(iterator + ": Solution after removing aisles: " + solution + " is feasible: "
                    + isSolutionFeasible(solution, false)); */
            return true;
        }

        int getNextRemovableAisle(RVNDSolution solution) {
            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(new HashSet<>(solution.orders()),
                    new HashSet<>(solution.aisles()));

            for (int i = 0; i < solution.aisles().size(); i++) {

                int aisleIndex = solution.aisles().get(i);
                HashMap<Integer, Integer> aisleItems = new HashMap<>(aisles.get(aisleIndex));
                boolean canRemove = true;

                for (Map.Entry<Integer, Integer> entry : aisleItems.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    if (itemsLeftInAisles.getOrDefault(item, 0) < quantity) {
                        canRemove = false;
                        break;
                    }
                }

                if (canRemove)
                    return i;
            }
            return -1;
        }
    }

    class AddOrder extends Neighborhood {
        ArrayList<Integer> candidateOrders;

        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality) {
            /* System.out.println("Exploring Add Order Neighborhood"); */
            RVNDSolution bestSolution = initialSolution;

            this.candidateOrders = IntStream.range(0, orders.size())
                    .boxed()
                    .parallel()
                    .filter(order -> !initialSolution.orders().contains(order))
                    .collect(Collectors.toCollection(ArrayList::new));

            Collections.shuffle(candidateOrders);
            /* System.out.println("Candidate Orders: " + candidateOrders); */

           /*  System.out.println("Initial Solution: " + initialSolution); */

            bestSolution = IntStream.range(0, candidateOrders.size())
                    .limit(this.MAX_ITERATIONS)
                    .parallel()
                    .mapToObj(orderIndex -> {
                        RVNDSolution currentSolution = new RVNDSolution(
                                new ArrayList<>(initialSolution.orders()),
                                new ArrayList<>(initialSolution.aisles()));

                        if (!this.move(currentSolution, orderIndex))
                            return null;

                        double currentQuality = computeObjectiveFunction(decodeRVNDSolution(currentSolution));
                        return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
                    })
                    .filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey)
                    .orElse(bestSolution);

           /*  System.out.println("Finished Exploring Add Order Neighborhood " + bestSolution); */
            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution, int iterator) {
            /* System.out.println(iterator + ": Performing a move in Add Order Neighborhood " + iterator); */
            if (iterator >= candidateOrders.size())
                return false;

            solution.orders().add(candidateOrders.get(iterator));

           /*  System.out.println(iterator + ": Solution after adding order " + iterator + ": " + solution + " is feasible: "
                    + isSolutionFeasible(solution, false)); */

            if (!isSolutionFeasible(solution, false))
                return false;

            int nextCandidateOrder = getNextCandidateOrderToAdd(solution);
            while (nextCandidateOrder != -1) {
                /* System.out.println(iterator + ": More orders could be added, adding order " + nextCandidateOrder); */
                solution.orders().add(candidateOrders.get(nextCandidateOrder));
                nextCandidateOrder = getNextCandidateOrderToAdd(solution);
            }

            /* System.out.println(iterator + ": Solution after adding orders: " + solution + " is feasible: "
                    + isSolutionFeasible(solution, false)); */
            return true;
        }

        int getNextCandidateOrderToAdd(RVNDSolution solution) {
            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(new HashSet<>(solution.orders()),
                    new HashSet<>(solution.aisles()));

            int totalItems = solution.orders().stream().mapToInt(order -> orders.get(order).values().stream().mapToInt(Integer::intValue).sum()).sum();

            for (int i = 0; i < candidateOrders.size(); i++) {
                int orderIndex = candidateOrders.get(i);
                if (solution.orders().contains(orderIndex))
                    continue;
                HashMap<Integer, Integer> orderItems = new HashMap<>(orders.get(orderIndex));

                if(orderItems.values().stream().mapToInt(Integer::intValue).sum() + totalItems > waveSizeUB)
                    continue;

                boolean canAdd = true;

                for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    if (itemsLeftInAisles.getOrDefault(item, 0) < quantity) {
                        canAdd = false;
                        break;
                    }
                }

                if (canAdd)
                    return i;
            }

            return -1;
        }
    }

    class Shake extends Neighborhood {
        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality) {
            /* System.out.println("Exploring Shake Neighborhood"); */
            RVNDSolution bestSolution = initialSolution;

            /* System.out.println("Initial Solution: " + initialSolution); */
            bestSolution = IntStream.range(0, this.MAX_ITERATIONS).parallel().mapToObj((orderIndex) -> {
                RVNDSolution currentSolution = new RVNDSolution(
                        new ArrayList<>(initialSolution.orders()),
                        new ArrayList<>(initialSolution.aisles()));

                if (!this.move(currentSolution, orderIndex)) {
                    return null;
                }

                double currentQuality = computeObjectiveFunction(decodeRVNDSolution(currentSolution));

                return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
            }).filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey) // Extrai a solução
                    .orElse(bestSolution);

            /* System.out.println("Finished Exploring Shake Neighborhood"); */
            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution, int iterator) {
            /* System.out.println(iterator + ": Performing a move in Shake Neighborhood " + iterator); */
            Random rand = new Random();

            int ordersSize = solution.orders().size();
            int aislesSize = solution.aisles().size();

            int numberOfOrdersToRemove = ordersSize > 0 ? rand.nextInt((int) Math.max(3, ordersSize * 0.1)) : 0;
            int numberOfAislesToRemove = aislesSize > 0 ? rand.nextInt((int) Math.max(3, aislesSize * 0.1)) : 0;

            /* System.out.println(iterator + ": Removing " + numberOfOrdersToRemove + " orders and " + numberOfAislesToRemove
                    + " aisles"); */

            for (int i = 0; i < numberOfOrdersToRemove && !solution.orders().isEmpty(); i++) {
                int orderIndex = rand.nextInt(solution.orders().size());
                /* System.out.println(iterator + ": Removing order " + orderIndex); */
                solution.orders().remove(orderIndex);
            }

            for (int i = 0; i < numberOfAislesToRemove && !solution.aisles().isEmpty(); i++) {
                int aisleIndex = rand.nextInt(solution.aisles.size());
                /* System.out.println(iterator + ": Removing aisle " + aisleIndex); */
                solution.aisles().remove(aisleIndex);
            }

            ordersSize = solution.orders().size();
            aislesSize = solution.aisles().size();

            int numberOfOrdersToAdd = orders.size() - ordersSize > 0
                    ? rand.nextInt((int) Math.max(3, (orders.size() - ordersSize) * 0.1))
                    : 0;
            int numberOfAislesToAdd = aisles.size() - aislesSize > 0
                    ? rand.nextInt((int) Math.max(3, (aisles.size() - aislesSize) * 0.1))
                    : 0;

            /* System.out.println(iterator + ": Adding " + numberOfOrdersToAdd + " orders and " + numberOfAislesToAdd
                    + " aisles"); */
            for (int i = 0; i < numberOfOrdersToAdd; i++) {
                int orderIndex = rand.nextInt(orders.size());
                if (solution.orders().contains(orderIndex))
                    continue;
                /* System.out.println(iterator + ": Adding order " + orderIndex); */
                solution.orders().add(orderIndex);
            }

            for (int i = 0; i < numberOfAislesToAdd; i++) {
                int aisleIndex = rand.nextInt(aisles.size());
                if (solution.aisles().contains(aisleIndex))
                    continue;
                /* System.out.println(iterator + ": Adding aisle " + aisleIndex); */
                solution.aisles().add(aisleIndex);
            }

            /* System.out.println(iterator + ": Solution after shake: " + solution + " is feasible: "
                    + isSolutionFeasible(solution, false)); */
            return isSolutionFeasible(solution, false);
        }

    }

    class OneAisleMultipleOrders extends Neighborhood {
        List<Integer> remainingOrders;
        List<Integer> remainingAisles;

        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality) {
           /*  System.out.println("Exploring One Aisle Multiple Orders Neighborhood"); */
            RVNDSolution bestSolution = initialSolution;

            this.remainingAisles = IntStream.range(0, aisles.size())
                    .boxed()
                    .parallel()
                    .filter(aisle -> !initialSolution.aisles().contains(aisle))
                    .sorted(Comparator.comparingInt(
                            (Integer aisle) -> aisles.get(aisle).values().stream().mapToInt(Integer::intValue).sum())
                            .reversed())
                    .collect(Collectors.toList());

            /* System.out.println("Remaining Aisles: " + remainingAisles); */

            this.remainingOrders = IntStream.range(0, orders.size())
                    .boxed()
                    .parallel()
                    .filter(order -> !initialSolution.orders().contains(order))
                    .sorted(Comparator.comparingInt(
                            (Integer order) -> orders.get(order).values().stream().mapToInt(Integer::intValue).sum())
                            .reversed())
                    .collect(Collectors.toList());

            Collections.shuffle(remainingOrders);

            /* System.out.println("Remaining Orders: " + remainingOrders); */

            /* System.out.println("Initial Solution: " + initialSolution); */

            bestSolution = IntStream.range(0, remainingAisles.size()).parallel().mapToObj((aisleIndex) ->{
                RVNDSolution currentSolution = new RVNDSolution(
                        new ArrayList<>(initialSolution.orders()),
                        new ArrayList<>(initialSolution.aisles()));

                if (!this.move(currentSolution, aisleIndex)) {
                    return null;
                }

                double currentQuality = computeObjectiveFunction(decodeRVNDSolution(currentSolution));

                return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
            }).filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey) // Extrai a solução
                    .orElse(bestSolution);

            /* System.out.println("Finished Exploring One Aisle Multiple Orders Neighborhood"); */
            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution, int iterator) {
            /* System.out.println(iterator + ": Performing a move in One Aisle Multiple Orders Neighborhood " + iterator); */
            if (iterator >= this.remainingAisles.size())
                return false;

            solution.aisles().add(remainingAisles.get(iterator));

            /* System.out.println(iterator + ": Solution after adding aisle " + iterator + ": " + solution + " is feasible: "
                    + isSolutionFeasible(solution, false)); */

            int nextAddableOrder = getNextCandidateOrderToAdd(solution);

            while (nextAddableOrder != -1) {
                solution.orders().add(remainingOrders.get(nextAddableOrder));
                /* System.out.println(iterator + ": Adding another order " + nextAddableOrder); */
                nextAddableOrder = getNextCandidateOrderToAdd(solution);
            }

            /* System.out.println(iterator + ": Solution after adding orders: " + solution + " is feasible: "
                    + isSolutionFeasible(solution, false)); */
            return isSolutionFeasible(solution, false);
        }

        int getNextCandidateOrderToAdd(RVNDSolution solution) {
            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(new HashSet<>(solution.orders()),
                    new HashSet<>(solution.aisles()));

            int totalItems = solution.orders().stream().mapToInt(order -> orders.get(order).values().stream().mapToInt(Integer::intValue).sum()).sum();

            for (int i = 0; i < remainingOrders.size(); i++) {
                int orderIndex = remainingOrders.get(i);
                if (solution.orders().contains(orderIndex))
                    continue;
                HashMap<Integer, Integer> orderItems = new HashMap<>(orders.get(orderIndex));

                if(orderItems.values().stream().mapToInt(Integer::intValue).sum() + totalItems > waveSizeUB)
                    continue;

                boolean canAdd = true;

                for (Map.Entry<Integer, Integer> entry : orderItems.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    if (itemsLeftInAisles.getOrDefault(item, 0) < quantity) {
                        canAdd = false;
                        break;
                    }
                }

                if (canAdd)
                    return i;
            }

            return -1;
        }
    }

    class OneOrderMultipleAisles extends Neighborhood {
        @Override
        RVNDSolution explore(RVNDSolution initialSolution, double initialQuality) {
            /* System.out.println("Exploring One Order Multiple Aisles Neighborhood"); */
            RVNDSolution bestSolution = initialSolution;

            initialSolution.orders().sort(Comparator.comparingInt(order -> orders.get(order).values().stream().mapToInt(Integer::intValue).sum()));

            /* System.out.println("Initial Solution: " + initialSolution); */

            bestSolution = IntStream.range(0, initialSolution.orders().size()).limit(this.MAX_ITERATIONS).parallel().mapToObj((orderIndex) -> {
                RVNDSolution currentSolution = new RVNDSolution(
                        new ArrayList<>(initialSolution.orders()),
                        new ArrayList<>(initialSolution.aisles()));

                if (!this.move(currentSolution, orderIndex)) {
                    return null;
                }

                double currentQuality = computeObjectiveFunction(decodeRVNDSolution(currentSolution));

                return new AbstractMap.SimpleEntry<>(currentSolution, currentQuality);
            }).filter(Objects::nonNull)
                    .max(Comparator.comparingDouble(AbstractMap.SimpleEntry::getValue))
                    .map(AbstractMap.SimpleEntry::getKey)
                    .orElse(bestSolution);

            /* System.out.println("Finished Exploring One Order Multiple Aisles Neighborhood " + bestSolution); */
            return bestSolution;
        }

        @Override
        boolean move(RVNDSolution solution, int iterator) {
            /* System.out.println(iterator + ": Performing a move in One Order Multiple Aisles Neighborhood " + iterator); */
            if(iterator >= solution.orders().size())
                return false;

            solution.orders().remove(iterator);

            /* System.out.println(iterator + ": Solution after removing order " + iterator + ": " + solution + " is feasible: "
                    + isSolutionFeasible(solution, false));
 */
            int nextRemovableAisle = getNextRemovableAisle(solution);

            while (nextRemovableAisle != -1) {
                solution.aisles().remove(nextRemovableAisle);
                /* System.out.println(iterator + ": Removing another aisle " + nextRemovableAisle); */
                nextRemovableAisle = getNextRemovableAisle(solution);
            }

            /* System.out.println(iterator + ": Solution after removing aisles: " + solution + " is feasible: "
                    + isSolutionFeasible(solution, false)); */
            return isSolutionFeasible(solution, false);
        }

        int getNextRemovableAisle(RVNDSolution solution) {
            HashMap<Integer, Integer> itemsLeftInAisles = getItemsLeftInAisles(new HashSet<>(solution.orders()),
                    new HashSet<>(solution.aisles()));

            for (int i = 0; i < solution.aisles().size(); i++) {

                int aisleIndex = solution.aisles().get(i);
                HashMap<Integer, Integer> aisleItems = new HashMap<>(aisles.get(aisleIndex));
                boolean canRemove = true;

                for (Map.Entry<Integer, Integer> entry : aisleItems.entrySet()) {
                    int item = entry.getKey();
                    int quantity = entry.getValue();
                    if (itemsLeftInAisles.getOrDefault(item, 0) < quantity) {
                        canRemove = false;
                        break;
                    }
                }

                if (canRemove)
                    return i;
            }
            return -1;
        }
    }

    private RVNDSolution randomVariableNeighborhoodDescent(RVNDSolution solution, double quality) {
        RVNDSolution bestSolution = solution;
        double bestQuality = quality;
        
        Collections.shuffle(solution.orders());
        Collections.shuffle(solution.aisles());

        ArrayList<Neighborhood> neighborhoods = new ArrayList<>();
        neighborhoods.add(new RemoveAisle());
        neighborhoods.add(new AddOrder());
        neighborhoods.add(new Shake());
        neighborhoods.add(new OneAisleMultipleOrders());
        neighborhoods.add(new OneOrderMultipleAisles());

        Collections.shuffle(neighborhoods);

        for (int i = 0; i < 3; i++) {
            Neighborhood neighborhood = neighborhoods.get(i);
            RVNDSolution currentSolution = neighborhood.explore(bestSolution, bestQuality);
            double currentQuality = computeObjectiveFunction(decodeRVNDSolution(currentSolution));

            if (currentQuality > bestQuality) {
                bestQuality = currentQuality;
                bestSolution = currentSolution;
                /* System.out.println("New best solution found by neighborhood exploration: " + bestQuality
                        + " current neighborhood: " + neighborhood.getClass().getName()); */
            }
        }

        return bestSolution;
    }

    protected ChallengeSolution decodeRVNDSolution(RVNDSolution solution) {
        return new ChallengeSolution(new HashSet<>(solution.orders()), new HashSet<>(solution.aisles));
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
