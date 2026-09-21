package Puzzle8;

import java.util.ArrayList;
import java.util.List;

public class NodoPuzzle {

    private String state;
    private NodoPuzzle parent;
    private int depth;
    private int cost;

    public NodoPuzzle(String state, NodoPuzzle parent) {

        this.state = state;
        this.parent = parent;

        if (parent != null) {
            depth = parent.getDepth() + 1;
            cost = parent.getCost() + 1;
        } else {
            depth = 0;
            cost = 0;
        }
    }

    public List<NodoPuzzle> generateChildren() {

        List<NodoPuzzle> successors = new ArrayList<>();

        int zeroPos = this.state.indexOf(" ");

        int[][] adjacentPositions = {
            {1, 3},
            {0, 2, 4},
            {1, 5},
            {0, 4, 6},
            {1, 3, 5, 7},
            {2, 4, 8},
            {3, 7},
            {4, 6, 8},
            {5, 7}
        };

        String newState;

        for (int adjPos : adjacentPositions[zeroPos]) {

            newState = swapPositions(
                    this.state,
                    zeroPos,
                    adjPos
            );

            successors.add(
                    new NodoPuzzle(newState, this)
            );
        }

        return successors;
    }

    private static String swapPositions(
            String state,
            int pos1,
            int pos2) {

        char[] arr = state.toCharArray();

        char temp = arr[pos1];

        arr[pos1] = arr[pos2];
        arr[pos2] = temp;

        return new String(arr);
    }

    public String getState() {
        return state;
    }

    public NodoPuzzle getParent() {
        return parent;
    }

    public int getDepth() {
        return depth;
    }

    public int getCost() {
        return cost;
    }

    @Override
    public String toString() {
        return state;
    }
}