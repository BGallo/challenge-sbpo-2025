package org.sbpo2025.challenge;

import org.apache.commons.lang3.time.StopWatch;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.TimeUnit;

import com.gurobi.gurobi.*;

public class ChallengeSolver {
    private final long MAX_RUNTIME = 60; // milliseconds; 10 minutes

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

    public ChallengeSolution solve(StopWatch stopWatch) throws GRBException {
        GRBEnv env = initializeGrbEnv();
        if (env == null) {
            return null;
        }

        ChallengeSolution solution = null;

        try {
            GRBModel model = new GRBModel(env);

            //variaveis x e y originais para pedidos e corredores
            GRBVar[] orderVars = new GRBVar[orders.size()];
            for (int i = 0; i < orders.size(); i++) {
                orderVars[i] = model.addVar(0, 1, 0, GRB.BINARY, "x_" + i);
            }

            GRBVar[] aisleVars = new GRBVar[aisles.size()];
            for (int i = 0; i < aisles.size(); i++) {
                aisleVars[i] = model.addVar(0, 1, 0, GRB.BINARY, "y_" + i);
            }

            //variaveis que decidem qual valor o denominador assume
            GRBVar[] dVars = new GRBVar[aisles.size()];

            GRBVar[] NdVars = new GRBVar[aisles.size()];

            for (int i = 0; i < aisles.size(); i++) {
                dVars[i] = model.addVar(0, 1, 0, GRB.BINARY, "d_" + (i));
                NdVars[i] = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "Nd_" + (i));
            }

            //restrição que obriga apenas um pedaço do denominador estar ativado
            GRBLinExpr expr = new GRBLinExpr();
            for (int i = 0; i < aisles.size(); i++) {
                expr.addTerm(1.0, dVars[i]);
            }

            model.addConstr(expr, GRB.EQUAL, 1.0, "one_denom_active");
            
            //restrição que liga o denominador com o número de corredores visitados
            GRBLinExpr sumY = new GRBLinExpr();
            GRBLinExpr denomByPiece = new GRBLinExpr();

            for (int i = 0; i < aisles.size(); i++) {
                sumY.addTerm(1.0, aisleVars[i]);
                denomByPiece.addTerm(i + 1.0, dVars[i]);
            }

            model.addConstr(sumY, GRB.EQUAL, denomByPiece, "denominator_link");


            int nMax = orders.stream()
              .flatMap(map -> map.values().stream())
              .mapToInt(Integer::intValue)
              .sum();

            int M = Math.min(nMax, waveSizeUB);

            for (int i = 0; i < aisles.size(); i++) {
                //lowerbound
                GRBLinExpr lowerBound = new GRBLinExpr();
                lowerBound.addTerm(waveSizeLB, dVars[i]);
                model.addConstr(NdVars[i], GRB.GREATER_EQUAL, lowerBound, "lower_bound_" + i);

                //upperbound
                GRBLinExpr upperBound = new GRBLinExpr();
                upperBound.addTerm(M, dVars[i]);
                model.addConstr(NdVars[i], GRB.LESS_EQUAL, upperBound, "upper_bound_" + i);
            }

            //Restrições quanto ao limite máximo e mínimo
            GRBLinExpr pickedItemsSum = new GRBLinExpr(); 

            for (int i = 0; i < orderVars.length; i++) {
                int sum = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();

                pickedItemsSum.addTerm(sum, orderVars[i]);
            }

            GRBLinExpr NdSum = new GRBLinExpr();
            for (int i = 0; i < aisles.size(); i++) {
                NdSum.addTerm(1.0, NdVars[i]);
            }

            model.addConstr(NdSum, GRB.EQUAL, pickedItemsSum, "Nd_def");

            //Restrição quanto a disponibilidade de items
            for (int item = 0; item < nItems; item++) {
                GRBLinExpr pickedItemExpr = new GRBLinExpr();
                GRBLinExpr availableItemExpr = new GRBLinExpr();

                for (int order = 0; order < orders.size(); order++) {
                    int quantityInOrder = orders.get(order).getOrDefault(item, 0);
                    if (quantityInOrder > 0) {
                        pickedItemExpr.addTerm(quantityInOrder, orderVars[order]);
                    }
                }

                for (int aisle = 0; aisle < aisles.size(); aisle++) {
                    int quantityInAisle = aisles.get(aisle).getOrDefault(item, 0);
                    if (quantityInAisle > 0) {
                        availableItemExpr.addTerm(quantityInAisle, aisleVars[aisle]);
                    }
                }

                model.addConstr(pickedItemExpr, GRB.LESS_EQUAL, availableItemExpr, "item_availability_" + item);
            }

            //Função objetivo
            GRBLinExpr obj = new GRBLinExpr();

            for (int i = 0; i < aisles.size(); i++) {
                obj.addTerm((1.0 / (i + 1.0)), NdVars[i]);
            }

            model.setObjective(obj, GRB.MAXIMIZE);

            model.optimize();

            if (model.get(GRB.IntAttr.SolCount) > 0) {
                Set<Integer> selectedOrders = new HashSet<>();
                Set<Integer> selectedAisles = new HashSet<>();

                for (int i = 0; i < orders.size(); i++) {
                    if (orderVars[i].get(GRB.DoubleAttr.X) > 0.5) {
                        selectedOrders.add(i);
                    }
                }

                for (int i = 0; i < aisles.size(); i++) {
                    if (aisleVars[i].get(GRB.DoubleAttr.X) > 0.5) {
                        selectedAisles.add(i);
                    }
                }

                solution = new ChallengeSolution(selectedOrders, selectedAisles);

                System.out.println("Objetivo ótimo: " + model.get(GRB.DoubleAttr.ObjVal));
            } else {
                System.out.println("Solução ótima não encontrada.");
            }

            model.dispose();
        } finally {
            env.dispose();
        }

        System.out.println("Solution is Feasible? " + isSolutionFeasible(solution));
        System.out.println("Objective Value: " + computeObjectiveFunction(solution));
        return solution;
    }

    private GRBEnv initializeGrbEnv() {
        try {
            GRBEnv env = new GRBEnv(true);
            env.set("logFile", "mip1.log");
            env.set(GRB.DoubleParam.TimeLimit, MAX_RUNTIME);
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