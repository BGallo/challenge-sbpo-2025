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

    public ChallengeSolution solve(StopWatch stopWatch) throws GRBException {
        GRBEnv env = initializeGrbEnv();
        if (env == null) {
            return null;
        }

        ChallengeSolution solution = null;

        try {
            GRBModel model = new GRBModel(env);

            //initializing variables
            GRBVar[] orderVars = new GRBVar[orders.size()];
            for (int i = 0; i < orders.size(); i++) {
                orderVars[i] = model.addVar(0, 1, 0, GRB.BINARY, "x_" + i);
            }

            GRBVar[] aisleVars = new GRBVar[aisles.size()];
            for (int i = 0; i < aisles.size(); i++) {
                aisleVars[i] = model.addVar(0, 1, 0, GRB.BINARY, "y_" + i);
            }

            //continuos variable for numerator
            GRBVar N = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "N");

            //continuous variable for denominator
            GRBVar D = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "D");

            //Adding constraingts that ensure N and D are their respective sums
            GRBLinExpr N_expr = new GRBLinExpr();
            for (int i = 0; i < orders.size(); i++) {
                int totalItems = orders.get(i).values().stream().mapToInt(Integer::intValue).sum();
                N_expr.addTerm(totalItems, orderVars[i]);
            }
            model.addConstr(N, GRB.EQUAL, N_expr, "Quanto ao valor de N");

            GRBLinExpr D_expr = new GRBLinExpr();
            for (GRBVar y : aisleVars) {
                D_expr.addTerm(1.0, y);
            }
            model.addConstr(D, GRB.EQUAL, D_expr, "Quanto ao valor de D");

            //piecewise variables
            int nPieces = aisleVars.length;

            //every possible configuration of D
            GRBVar[] dVars = new GRBVar[nPieces];

            //every possible configuration of N *  d_i
            GRBVar[] NdVars = new GRBVar[nPieces];

            for (int i = 0; i < nPieces; i++) {
                dVars[i] = model.addVar(0, 1, 0, GRB.BINARY, "d_" + (i));
                NdVars[i] = model.addVar(0, GRB.INFINITY, 0, GRB.CONTINUOUS, "Nd_" + (i));
            }
            
            //maximun value of N
            int nMax = orders.stream()
              .flatMap(map -> map.values().stream())
              .mapToInt(Integer::intValue)
              .sum();

            int M = Math.min(nMax, waveSizeUB);

            for (int i = 0; i < nPieces; i++) {
                // Nd_i <= N
                model.addConstr(NdVars[i], GRB.LESS_EQUAL, N, "c1_Nd_" + i);

                // Nd_i <= nMax * d_i
                GRBLinExpr upperNdBound = new GRBLinExpr();
                upperNdBound.addTerm(1.0, NdVars[i]);
                upperNdBound.addTerm(-M, dVars[i]);
                model.addConstr(upperNdBound, GRB.LESS_EQUAL, 0.0, "c2_Nd_" + i);

                // Nd_i >= N - M*(1 - d_i)
                GRBLinExpr lowerNdBound = new GRBLinExpr();
                lowerNdBound.addTerm(1.0, NdVars[i]);
                lowerNdBound.addTerm(-1.0, N);
                lowerNdBound.addTerm(M, dVars[i]);
                model.addConstr(lowerNdBound, GRB.GREATER_EQUAL, -M, "c3_Nd_" + i);

                // Nd_i >= 0
                model.addConstr(NdVars[i], GRB.GREATER_EQUAL, 0.0, "c4_Nd_" + i);
            }

            GRBLinExpr D_byPiece = new GRBLinExpr();
            for (int i = 0; i < nPieces; i++) {
                D_byPiece.addTerm(i + 1.0, dVars[i]);
            }
            model.addConstr(D, GRB.EQUAL, D_byPiece, "D_piecewise");

            GRBLinExpr currentPiece = new GRBLinExpr();
            for (GRBVar dv : dVars) currentPiece.addTerm(1.0, dv);
            model.addConstr(currentPiece, GRB.EQUAL, 1, "onePieceActive");

            //Para cada Item I
            for (int i = 0; i < nItems; i++) {

                //O número desse item nos pedidos
                GRBLinExpr orderItems = new GRBLinExpr();
                for (int p = 0; p < orders.size(); p++) {
                    int qty = orders.get(p).getOrDefault(i, 0);
                    if (qty > 0) orderItems.addTerm(qty, orderVars[p]);
                }

                //Deve ser menor igual ao número desse item nos corredores
                GRBLinExpr aislesItems = new GRBLinExpr();
                for (int c = 0; c < aisles.size(); c++) {
                    int qty = aisles.get(c).getOrDefault(i, 0);
                    if (qty > 0) aislesItems.addTerm(qty, aisleVars[c]);
                }

                model.addConstr(orderItems, GRB.LESS_EQUAL, aislesItems, "cover_item_" + i);
            }

            model.addConstr(N, GRB.GREATER_EQUAL, waveSizeLB, "LI");
            model.addConstr(N, GRB.LESS_EQUAL, waveSizeUB, "LS");

            GRBLinExpr obj = new GRBLinExpr();
            for (int i = 0; i < nPieces; i++) {
                obj.addTerm(1.0 / (i + 1), NdVars[i]);
            }
            model.setObjective(obj, GRB.MAXIMIZE);

            model.optimize();

            if (model.get(GRB.IntAttr.Status) == GRB.Status.OPTIMAL) {
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