package Puzzle8;
public class Utils {
    
    public static void formatState(String state){
        String formattedState = "";

        for(int i = 0; i < state.length(); i++){
            formattedState += state.charAt(i) + " ";
            if((i + 1) % 3 == 0){
                formattedState += "\n";
            }
        }

        System.out.println(formattedState);
    }

}
