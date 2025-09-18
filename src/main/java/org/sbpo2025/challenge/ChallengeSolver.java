package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import com.gurobi.gurobi.*;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 60; // seconds

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

    public ChallengeSolution solve(StopWatch stopWatch, int nThreads) throws GRBException {
        GRBEnv env = initializeGrbEnv(stopWatch, nThreads);
        if (env == null) {
            return null;
        }


        ChallengeSolution solution = getInitialViableSolution();

        try{
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

            Double lambda = computeObjectiveFunction(solution);

            int iteration = 0;
            while (stopWatch.getTime(TimeUnit.SECONDS) < MAX_RUNTIME) {
                iteration++;

                for (int p = 0; p < orders.size(); p++) {
                    x[p].set(GRB.DoubleAttr.Start, solution.orders().contains(p) ? 1.0 : 0.0);
                }

                for (int c = 0; c < aisles.size(); c++) {
                    y[c].set(GRB.DoubleAttr.Start, solution.aisles().contains(c) ? 1.0 : 0.0);
                }

                //Parte de cima da fração
                GRBLinExpr obj = new GRBLinExpr();
                for (int p = 0; p < orders.size(); p++) {
                    int totalItemsInOrder = orders.get(p).values().stream().mapToInt(Integer::intValue).sum();
                    obj.addTerm(totalItemsInOrder, x[p]);
                }

                //Parte de baixo da fração
                for (int c = 0; c < aisles.size(); c++) {
                    obj.addTerm(-lambda, y[c]);
                }

                model.setObjective(obj, GRB.MAXIMIZE);

                model.optimize();

                if (model.get(GRB.IntAttr.SolCount) > 0) {

                    solution = convertToChallengeSolution(x, y);

                    if (model.get(GRB.DoubleAttr.ObjVal) < 1e-6) {
                        break;
                    }

                    lambda = computeObjectiveFunction(solution);

                    System.out.println("Iteração: " + iteration + " - lambda: " + lambda);

                    if (MAX_RUNTIME - stopWatch.getTime(TimeUnit.SECONDS) < 0.5) {
                        break;
                    }

                    model.getEnv().set(GRB.DoubleParam.TimeLimit, Math.min(MAX_RUNTIME - stopWatch.getTime(TimeUnit.SECONDS), MAX_RUNTIME/2));

                } else {
                    break;
                }

            }

            model.dispose();

        } finally {
            env.dispose();
        }

        System.out.println("Solution is Feasible? " + isSolutionFeasible(solution));
        System.out.println("Objective Value: " + computeObjectiveFunction(solution));
        return solution;
    }

    private ChallengeSolution convertToChallengeSolution(GRBVar[] orderVars, GRBVar[] aisleVars) throws GRBException {
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();

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

        return new ChallengeSolution(selectedOrders, selectedAisles);
    }

    private ChallengeSolution getInitialViableSolution() {
        Set<Integer> selectedOrders = new HashSet<>();
        Set<Integer> selectedAisles = new HashSet<>();
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

        return new ChallengeSolution(selectedOrders, selectedAisles);
    }

    private GRBEnv initializeGrbEnv(StopWatch stopWatch, int nThreads) {
        try {
            GRBEnv env = new GRBEnv(true);
            env.set("logFile", "mip1.log");
            env.set(GRB.DoubleParam.TimeLimit, Math.min(MAX_RUNTIME - stopWatch.getTime(TimeUnit.SECONDS), MAX_RUNTIME/2));
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
            System.out.println("Total units picked out of bounds");
            return false;
        }

        // Check if the units picked do not exceed the units available
        for (int i = 0; i < nItems; i++) {
            if (totalUnitsPicked[i] > totalUnitsAvailable[i]) {
                System.out.println("Total units picked exceed available units");
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