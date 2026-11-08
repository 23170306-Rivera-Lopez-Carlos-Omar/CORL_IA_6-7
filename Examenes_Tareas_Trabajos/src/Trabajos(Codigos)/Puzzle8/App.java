

public class App {
    public static void main(String[] args) {



        String initialState = "12345 678"; //random inital state
        String goalState = "12345678"; //goal state
        SearchTree searchTree = new SearchTree(initialState, goalState);

        searchTree.breadFirstSearch();



        System.out.println("HelloWolrd");

        NodeUtils.generateChildren(new Node(initialState, null)).forEach(child -> System.out.println(child.getState()));
        for (Node child : NodeUtils.generateChildren(new Node(initialState, null))) {
            System.out.println(child.getState());


    }


    }}

