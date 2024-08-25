/*
* 1. Stairs
* 2. Fibonacci Number
* 3. Minimum Number of Squares
* 4. Maximum SUm Value
* 5. Max Product Subarray
* 6. Ways to Decode
* 7. Max Sum without Adjacent Elements
* 8. 2D Unique Paths in A Grid
* 9. Dungeon Princess
* 10. Maximum Subseq Sum
* 11. Min Sum Path in Matrix
* 12. Max Rectangle in Binary Matrix
* 13. Min Sum Path in Triangle
* 14. Knapsack - Fractional Knapsack
* 15. Knapsack - 0-1 Knapsack
* 16. Knapsack - Unbounded Knapsack
* 17. Knapsack - Buying Candies
* 18. Knapsack - Tushar's Birthday Party
* 19. Knapsack - Ways to send Signal
* 20. Applications of KNapsack - Cutting a Rod
* 21. Applications of KNapsack - Coin Sum Infinite
* 22. Applications of KNapsack - 0-1 KnapSack 2
* 23. Applications of KNapsack - Distinct Subsequences
* 24. Applications of KNapsack - Length of longest Fibonacci SubSequence
* 25. Applications of KNapsack - Let's Party
* */

// 1. Stairs
public class Solution {
    public int climbStairs(int A) {
        if(A <= 1){ return 1;}
        int MOD = 1000000007;
        int[] dp = new int[A + 1];
        dp[0] = 1;
        dp[1] = 1;
        for(int i = 2; i <= A; i++){
            dp[i] = (dp[i - 1] + dp[i - 2]) % MOD;
        }
        return dp[A];
    }
}

//* 2. Fibonacci Number

public static int solve(int A) {
    // Base Case
    if(A == 0 || A==1){return A;}
    // Initialising DP Array
    int[] fib = new int[A + 1];
    //Initialisation for the firsttwo elements
    fib[0] = 0;
    fib[1] = 1;
    //Dynamic Programming Transition
    for(int i = 2; i <= A; i++){
        fib[i] = fib[i - 1] + fib[i - 2];
    }
    return fib[A];
}
//* 3. Minimum Number of Squares

public class Solution {
    public int countMinSquares(int A) {
        //Base Case
        if(A <= 0){return 0;}
        // Dynamic Programming Array
        int[] dp = new int[A + 1];
        //Initialisation of dp of i to max value
        for(int i = 1; i <= A; i++){dp[i] = Integer.MAX_VALUE;}
        //Base Case for DP array
        dp[0] = 0;
        for(int i = 1; i <= A; i++){
            for(int j = 1; j*j <= i; j++){
                dp[i] = Math.min(dp[i], dp[i - j*j] + 1);
            }
        }
        return dp[A];
    }
}


//* 4. Maximum SUm Value
public class Solution {
    public int solve(int[] A, int B, int C, int D) {
        int N = A.length;
        //Arrays for DP
        int[] left = new int[N];
        int[] middle = new int[N];
        int[] right = new int[N];
        //1st Stage
        left[0] = A[0] * B;
        for(int i = 1; i < N; i++){
            left[i] = Math.max(left[i - 1], A[i] * B);
        }
        //2ndStage
        middle[0] = left[0] + A[0] * C;
        for(int i = 1; i < N; i++){
            middle[i] = Math.max(middle[i-1], left[i] + A[i] * C);
        }
        //3rd Stage
        right[0] = middle[0] + A[0] * D;
        for(int i = 1; i < N; i++){
            right[i] = Math.max(right[i-1], middle[i] + A[i] * D);
        }
        return right[N -1];
    }
}



//* 5. Max Product Subarray


public class Solution {
    // DO NOT MODIFY THE ARGUMENTS WITH "final" PREFIX. IT IS READ ONLY
    public int maxProduct(final int[] A) {
        // Edge case: If the array is empty, return 0
        if (A == null || A.length == 0) {
            return 0;
        }

        int N = A.length;
        int[] maxDP = new int[N];
        int[] minDP = new int[N];

        // Initialize the first element
        maxDP[0] = A[0];
        minDP[0] = A[0];

        int maxProduct = A[0];  // Initialize the result to the first element

        // Fill in the DP arrays
        for (int i = 1; i < N; i++) {
            if (A[i] > 0) {
                maxDP[i] = Math.max(A[i], maxDP[i-1] * A[i]);
                minDP[i] = Math.min(A[i], minDP[i-1] * A[i]);
            } else {
                maxDP[i] = Math.max(A[i], minDP[i-1] * A[i]);
                minDP[i] = Math.min(A[i], maxDP[i-1] * A[i]);
            }

            // Update the global maximum product
            maxProduct = Math.max(maxProduct, maxDP[i]);
        }

        return maxProduct;
    }
}



//* 6. Ways to Decode

public class Solution {
    public int numDecodings(String A) {
        //Edge Case Handling
        int MOD = 1000000007;
        int n = A.length();
        if(n == 0 || A.charAt(0) == '0'){return 0;}
        //DP Array Initialisation
        int[] dp = new int[n+1];
        dp[0] = 1;
        dp[1] = 1;
        //Populating the DP Array
        for(int i = 2; i <= n; i++){
            //One digit decode
            int oneDigit = Integer.parseInt(A.substring(i - 1, i));
            if(oneDigit >= 1 && oneDigit <= 9){
                dp[i] += dp[i-1];
                dp[i] %= MOD;
            }

            //Two digit decode
            int twoDigits = Integer.parseInt(A.substring(i - 2, i));
            if(twoDigits >= 10 && twoDigits <= 26){
                dp[i] += dp[i -2];
                dp[i] %= MOD;
            }
        }
        return dp[n];
    }
}



//* 7. Max Sum without Adjacent Elements


public class Solution {
    public int adjacent(int[][] A) {
        //Edge Cases
        int N = A[0].length;
        if(N == 0){return 0;}
        if(N == 1){return Math.max(A[0][0], A[1][0]);}
        //DP Array Initialisation
        int[] dp = new int[N];
        dp[0] = Math.max(A[0][0], A[1][0]);
        dp[1] = Math.max(A[0][1], A[1][1]);
        dp[1] = Math.max(dp[0], dp[1]);
        //Populating the DP Array
        for(int i = 2; i < N; i++){
            int currentMax = Math.max(A[0][i], A[1][i]);
            dp[i] = Math.max(dp[i - 1], dp[i - 2]+ currentMax);
        }
        return dp[N - 1];
    }
}


//8. 2D Unique Paths in A Grid

public class Solution {
    public int uniquePathsWithObstacles(int[][] A) {
        int n = A.length;
        int m = A[0].length;
        if (A[0][0] == 1 || A[n-1][m-1] == 1) {
            return 0;
        }
        int[] dp = new int[m];
        dp[0] = 1; // Starting point
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (A[i][j] == 1) {
                    dp[j] = 0;
                } else if (j > 0) {
                    dp[j] += dp[j-1];
                }
            }
        }
        return dp[m-1];
    }
}



//* 9. Dungeon Princess

public class Solution {
    public int calculateMinimumHP(int[][] A) {
        int M = A.length;
        int N = A[0].length;
        int[][] dp = new int[M][N];
        dp[M-1][N-1] = Math.max(1, 1 - A[M-1][N-1]);

        for (int i = M-2; i >= 0; i--) {
            dp[i][N-1] = Math.max(1, dp[i+1][N-1] - A[i][N-1]);
        }

        for (int j = N-2; j >= 0; j--) {
            dp[M-1][j] = Math.max(1, dp[M-1][j+1] - A[M-1][j]);
        }
        for (int i = M-2; i >= 0; i--) {
            for (int j = N-2; j >= 0; j--) {
                dp[i][j] = Math.max(1, Math.min(dp[i+1][j], dp[i][j+1]) - A[i][j]);
            }
        }
        return dp[0][0];
    }
}



//* 10. Maximum Subseq Sum

public class Solution {
    public int maxSubsequenceSum(int[] A) {
        int N = A.length;
        if (N == 0) return 0;
        if (N == 1) return A[0];

        int[] dp = new int[N];

        dp[0] = A[0];
        dp[1] = Math.max(A[0], A[1]);

        for (int i = 2; i < N; i++) {
            dp[i] = Math.max(dp[i-1], A[i] + dp[i-2]);
        }

        return dp[N-1];
    }
}




//* 11. Min Sum Path in Matrix

public class Solution {
    public int minPathSum(int[][] A) {
        int M = A.length;
        int N = A[0].length;

        int[][] dp = new int[M][N];

        dp[0][0] = A[0][0];

        for (int j = 1; j < N; j++) {
            dp[0][j] = dp[0][j-1] + A[0][j];
        }

        for (int i = 1; i < M; i++) {
            dp[i][0] = dp[i-1][0] + A[i][0];
        }

        for (int i = 1; i < M; i++) {
            for (int j = 1; j < N; j++) {
                dp[i][j] = Math.min(dp[i-1][j], dp[i][j-1]) + A[i][j];
            }
        }

        return dp[M-1][N-1];
    }
}


//* 12. Max Rectangle in Binary Matrix

public class Solution {
    public int maximalRectangle(int[][] A) {
        if (A.length == 0) return 0;

        int N = A.length;
        int M = A[0].length;
        int[] heights = new int[M];
        int maxArea = 0;

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < M; j++) {

                if (A[i][j] == 1) {
                    heights[j]++;
                } else {
                    heights[j] = 0;
                }
            }

            maxArea = Math.max(maxArea, largestRectangleArea(heights));
        }

        return maxArea;
    }

    private int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        int N = heights.length;

        for (int i = 0; i <= N; i++) {
            int height = (i == N) ? 0 : heights[i];
            while (!stack.isEmpty() && height < heights[stack.peek()]) {
                int h = heights[stack.pop()];
                int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                maxArea = Math.max(maxArea, h * width);
            }
            stack.push(i);
        }

        return maxArea;
    }
}



//* 13. Min Sum Path in Triangle

public class Solution {
    public int minimumTotal(ArrayList<ArrayList<Integer>> A) {
        int n = A.size();

        if (n == 0) return 0;

        int[] dp = new int[n];

        for (int i = 0; i < n; i++) {
            dp[i] = A.get(n - 1).get(i);
        }

        for (int i = n - 2; i >= 0; i--) {
            for (int j = 0; j <= i; j++) {
                dp[j] = A.get(i).get(j) + Math.min(dp[j], dp[j + 1]);
            }
        }
        return dp[0];
    }
}




//14. Knapsack - Fractional Knapsack

public class Solution {
    static class Item implements Comparable<Item>{
        int value, weight;
        double valuePerWeight;

        Item(int value, int weight){
            this.value = value;
            this.weight = weight;
            this.valuePerWeight = (double) value/ weight;

        }
        @Override
        public int compareTo(Item other){
            return Double.compare(other.valuePerWeight, this.valuePerWeight);
        }
    }
    public int solve(int[] A, int[] B, int C){
        int n = A.length;
        Item[] items = new Item[n];
        for(int i = 0; i < n; i++){
            A[i] *= 100;
            items[i] = new Item(A[i], B[i]);
        }
        Arrays.sort(items);
        int maxValue = 0;
        int currentCapacity = C;
        for(Item item : items){
            if(currentCapacity >= item.weight){
                maxValue += item.value;
                currentCapacity -= item.weight;
            }
            else{
                maxValue +=(item.valuePerWeight * currentCapacity);
                break;
            }
        }
        return maxValue;

    }
}



//* 15. Knapsack - 0-1 Knapsack

public class Solution {
    public int solve(int[] A, int[] B, int C) {
        int N = A.length;
        int[][] dp = new int[N + 1][C + 1];
        for(int i = 1; i <= N; i++){
            for(int j = 1; j <= C; j++){
                dp[i][j]=dp[i-1][j];
                if(B[i-1] <= j){
                    dp[i][j]= Math.max(dp[i][j], dp[i-1][j-B[i-1]] + A[i-1]);
                }
            }
        }
        return dp[N][C];
    }
}




//* 16. Knapsack - Unbounded Knapsack

public class Solution {
    public int solve(int A, int[] B, int[] C) {
        int[] dp = new int[A + 1];
        for(int i = 0; i < B.length; i++){
            for(int j = C[i]; j <= A; j++){
                dp[j] = Math.max(dp[j], dp[j - C[i]] + B[i]);
            }
        }
        return dp[A];
    }
}





//* 17. Knapsack - Buying Candies (Left)


//* 18. Knapsack - Tushar's Birthday Party

public class Solution {
    // DO NOT MODIFY THE ARGUMENTS WITH "final" PREFIX. IT IS READ ONLY
    public int solve(final int[] A, final int[] B, final int[] C) {
        int N = A.length;
        int M = B.length;

        int maxCapacity = 0;
        for (int i = 0; i < N; i++) {
            maxCapacity = Math.max(maxCapacity, A[i]);
        }

        int[] dp = new int[maxCapacity + 1];
        for (int i = 1; i <= maxCapacity; i++) {
            dp[i] = Integer.MAX_VALUE;
        }
        dp[0] = 0;

        for (int i = 1; i <= maxCapacity; i++) {
            for (int j = 0; j < M; j++) {
                if (i >= B[j] && dp[i - B[j]] != Integer.MAX_VALUE) {
                    dp[i] = Math.min(dp[i], dp[i - B[j]] + C[j]);
                }
            }
        }

        int totalCost = 0;
        for (int i = 0; i < N; i++) {
            totalCost += dp[A[i]];
        }

        return totalCost;
    }
}




//* 19. Knapsack - Ways to send Signal


public class Solution {
    public int solve(int A) {

        final int MOD = 1000000007;

        if (A == 0) return 1;
        if (A == 1) return 2;

        int[] dp = new int[A + 1];
        dp[0] = 1;
        dp[1] = 2;
        for (int i = 2; i <= A; i++) {
            dp[i] = (dp[i - 1] + dp[i - 2]) % MOD;
        }

        return dp[A];
    }
}



//20. Applications of KNapsack - Cutting a Rod

public class Solution {
    public int solve(int[] A) {
        int N = A.length;
        int[] dp = new int[N + 1];

        // Initialize dp array
        dp[0] = 0;

        // Fill the dp array
        for (int i = 1; i <= N; i++) {
            for (int j = 1; j <= i; j++) {
                dp[i] = Math.max(dp[i], A[j - 1] + dp[i - j]);
            }
        }

        return dp[N];
    }
}






//* 21. Applications of KNapsack - Coin Sum Infinite


public class Solution {
    private static final int MOD = 1000007;

    public int coinchange2(int[] A, int B) {
        int[] dp = new int[B + 1];
        dp[0] = 1;  // Base case: There's one way to make sum 0 (using no coins)

        for (int coin : A) {
            for (int j = coin; j <= B; j++) {
                dp[j] = (dp[j] + dp[j - coin]) % MOD;
            }
        }

        return dp[B];
    }
}



//* 22. Applications of KNapsack - 0-1 KnapSack 2


public class Solution {
    public int solve(int[] A, int[] B, int C) {
        int N = A.length;
        int[] dp = new int[C + 1];

        // Initialize dp array with 0s (default value)
        for (int i = 0; i <= C; i++) {
            dp[i] = 0;
        }

        // Fill the dp array
        for (int i = 0; i < N; i++) {
            int value = A[i];
            int weight = B[i];
            for (int j = C; j >= weight; j--) {
                dp[j] = Math.max(dp[j], dp[j - weight] + value);
            }
        }

        return dp[C];
    }
}



//* 23. Applications of KNapsack - Distinct Subsequences


public class Solution {
    public int numDistinct(String A, String B) {
        int n = A.length();
        int m = B.length();

        // Create a DP table
        int[][] dp = new int[n + 1][m + 1];

        // Initialize the first column of dp table
        for (int i = 0; i <= n; i++) {
            dp[i][0] = 1;
        }

        // Fill the DP table
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (A.charAt(i - 1) == B.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }

        // The result is the number of distinct ways to form B from A
        return dp[n][m];
    }
}




//* 24. Applications of KNapsack - Length of longest Fibonacci SubSequence

public class Solution {
    public int solve(int[] A) {
        int n = A.length;
        if (n < 3) return 0;

        // Create a map to store the indices of elements in A
        Map<Integer, Integer> indexMap = new HashMap<>();
        for (int i = 0; i < n; i++) {
            indexMap.put(A[i], i);
        }

        // DP table to store the length of the longest subsequence ending with A[i] and A[j]
        int[][] dp = new int[n][n];
        int maxLen = 0;

        // Iterate over pairs (i, j) with j > i
        for (int j = 1; j < n; j++) {
            for (int i = 0; i < j; i++) {
                int expected = A[j] - A[i];
                if (expected < A[i] && indexMap.containsKey(expected)) {
                    int k = indexMap.get(expected);
                    dp[i][j] = dp[k][i] + 1;
                    maxLen = Math.max(maxLen, dp[i][j] + 2); // +2 for the elements A[k] and A[i]
                }
            }
        }

        return maxLen >= 3 ? maxLen : 0;
    }
}




//* 25. Applications of KNapsack - Let's Party


public class Solution {
    public int solve(int A) {
        final int MOD = 10003;
        if (A == 0) return 1; // Base case for 0 people
        if (A == 1) return 1; // Base case for 1 person

        int[] dp = new int[A + 1];
        dp[0] = 1; // 1 way to arrange 0 people
        dp[1] = 1; // 1 way to arrange 1 person

        for (int i = 2; i <= A; i++) {
            dp[i] = (dp[i - 1] + (i - 1) * dp[i - 2]) % MOD;
        }

        return dp[A];
    }
}
