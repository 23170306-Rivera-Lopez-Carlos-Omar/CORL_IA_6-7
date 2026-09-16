package Puzzle8;

import java.util.ArrayList;
import java.util.List;

public class NodoPuzzle {
    private String state;
    private NodoPuzzle parent;
    private int depth;

    public NodoPuzzle(String state, NodoPuzzle parent) {
        this.state = state;
        this.parent = parent;
        if(parent != null){
            depth = parent.getDepth() + 1;
        } else {
            depth = 0;
        }
    }

    public List<NodoPuzzle> generateChildren() {
        // Usamos ArrayList estándar de Java
        List<NodoPuzzle> successors = new ArrayList<>();

        int zeroPos = this.state.indexOf(" ");

        int[][] adjacentPositions = {
            {1, 3},           // pos 0: can swap with right(1), down(3)
            {0, 2, 4},        // pos 1: can swap with left(0), right(2), down(4)
            {1, 5},           // pos 2: can swap with left(1), down(5)
            {0, 4, 6},        // pos 3: can swap with up(0), right(4), down(6)
            {1, 3, 5, 7},     // pos 4: can swap with up(1), left(3), right(5), down(7)
            {2, 4, 8},        // pos 5: can swap with up(2), left(4), down(8)
            {3, 7},           // pos 6: can swap with up(3), right(7)
            {4, 6, 8},        // pos 7: can swap with up(4), left(6), right(8)
            {5, 7}            // pos 8: can swap with up(5), left(7)
        };

        String newState;
        for (int adjPos : adjacentPositions[zeroPos]) {
            newState = swapPositions(this.state, zeroPos, adjPos);
            successors.add(new NodoPuzzle(newState, this)); // .add en lugar de InsertarFinal
        }

        return successors;
    }

    private static String swapPositions(String state, int pos1, int pos2) {
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

    public String toString(){
        return state;
    }
}