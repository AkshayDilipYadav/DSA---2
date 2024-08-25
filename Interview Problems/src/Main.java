/*
* 1. Length of longest Consecutive Ones
* 2. Majority Element
* 3. Row to Column Zero
* 4. N/3 Repeat Number
* 5. Check Anagrams
* 6. Colorful Number
* 7. Hashing - Shaggy and Distances
* 8. Hashing - Longest Subarray Zero Sum
* 9. Hashing - Sort Array in given Order
* 10. Hashing - Colorful Number
* 11. Hashing - Count Subarrays
* 12. Hashing - Longest Substring Without Repeat
* 13. Length of longest consecutive ones
* 14. Majority Element
* 15. N/3 Repeat Number
* 16. Check Anagrams
* 17. Colorful Number
* */

// 1. Length of longest Consecutive Ones

public class Solution {
    public int solve(String A) {
        int n = A.length();
        int totalOnes = 0;
        for(char c : A.toCharArray()){
            if(c == '1'){
                totalOnes++;
            }
        }

        if(totalOnes ==0){
            return 0;

        }
        int maxLength = 0;
        int currentLength = 0;
        int previousLength = -1;
        for(int i = 0; i < n; i++){
            if(A.charAt(i) == '1'){currentLength++;}
            else{
                if(previousLength != -1){
                    maxLength = Math.max(maxLength, previousLength + currentLength +1);
                }
                previousLength = currentLength;
                currentLength =0;
            }
            maxLength = Math.max(maxLength, currentLength +1);
        }
        maxLength = Math.max(maxLength, previousLength + currentLength +1);
        maxLength = Math.min(maxLength, totalOnes);
        return maxLength;
    }
}




//* 2. Majority Element

public class Solution {
    // DO NOT MODIFY THE ARGUMENTS WITH "final" PREFIX. IT IS READ ONLY
    public int majorityElement(final int[] A) {
        int n = A.length;
        int majorityElement = A[0];
        int count = 1;
        for (int i = 1; i < n; i++) {
            if (A[i] == majorityElement) {
                count++;
            } else {
                count--;
                if (count == 0) {
                    majorityElement = A[i];
                    count = 1;
                }
            }
        }
        count = 0;
        for (int i = 0; i < n; i++) {
            if (A[i] == majorityElement) {
                count++;
            }
        }

        if (count > n / 2) {
            return majorityElement;
        }
        return -1;

    }
}


//* 3. Row to Column Zero

public class Solution {
    public int[][] solve(int[][] A) {
        int m = A.length;
        int n = A[0].length;

        boolean[] zeroRows = new boolean[m];
        boolean[] zeroCols = new boolean[n];

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (A[i][j] == 0) {
                    zeroRows[i] = true;
                    zeroCols[j] = true;
                }
            }
        }
        for (int i = 0; i < m; i++) {
            if (zeroRows[i]) {
                for (int j = 0; j < n; j++) {
                    A[i][j] = 0;
                }
            }
        }
        for (int j = 0; j < n; j++) {
            if (zeroCols[j]) {
                for (int i = 0; i < m; i++) {
                    A[i][j] = 0;
                }
            }
        }
        return A;

    }
}


//* 4. N/3 Repeat Number


public class Solution {
    public int repeatedNumber(int[] A) {
        int n = A.length;

        if (n == 0) {
            return -1;
        }

        int candidate1 = Integer.MAX_VALUE, count1 = 0;
        int candidate2 = Integer.MAX_VALUE, count2 = 0;

        for (int i = 0; i < n; i++) {
            int num = A[i];

            if (num == candidate1) {
                count1++;
            } else if (num == candidate2) {
                count2++;
            } else if (count1 == 0) {
                candidate1 = num;
                count1 = 1;
            } else if (count2 == 0) {
                candidate2 = num;
                count2 = 1;
            } else {
                count1--;
                count2--;
            }
        }
        count1 = 0;
        count2 = 0;
        for (int i = 0; i < n; i++) {
            int num = A[i];

            if (num == candidate1) {
                count1++;
            } else if (num == candidate2) {
                count2++;
            }
        }

        if (count1 > n / 3) {
            return candidate1;
        } else if (count2 > n / 3) {
            return candidate2;
        }
        return -1;

    }
}



//* 5. Check Anagrams

public class Solution {
    public int solve(String A, String B) {
        if (A.length() != B.length()) {
            return 0;
        }char[] charArrayA = A.toCharArray();
        char[] charArrayB = B.toCharArray();
        Arrays.sort(charArrayA);
        Arrays.sort(charArrayB);
        if (Arrays.equals(charArrayA, charArrayB)) {
            return 1;
        } else {
            return 0;
        }

    }
}



//* 6. Colorful Number


public class Solution {
    public int colorful(int A) {
        String numStr = String.valueOf(A);
        int n = numStr.length();
        HashSet<Integer> products = new HashSet<>();
        for (int i = 0; i < n; i++) {
            int product = 1;
            for (int j = i; j < n; j++) {
                product *= Character.getNumericValue(numStr.charAt(j));
                if (products.contains(product)) {
                    return 0;
                }
                products.add(product);
            }
        }

        return 1;
    }
}


//7. Hashing - Shaggy and Distances

public class Solution {
    public int solve(int[] A) {
        HashMap<Integer, Integer> lastSeen = new HashMap<>();
        int minDistance = Integer.MAX_VALUE;
        boolean found = false;

        for (int i = 0; i < A.length; i++) {
            if (lastSeen.containsKey(A[i])) {
                int distance = i - lastSeen.get(A[i]);
                minDistance = Math.min(minDistance, distance);
                found = true;
            }
            lastSeen.put(A[i], i);
        }

        return found ? minDistance : -1;
    }
}



//* 8. Hashing - Longest Subarray Zero Sum

public class Solution {
    public int solve(int[] A) {
        HashMap<Long, Integer> prefixSumMap = new HashMap<>();
        long prefixSum = 0;
        int maxLength = 0;

        for (int i = 0; i < A.length; i++) {
            prefixSum += A[i];

            // If the prefix sum is zero, the subarray from the start to the current index sums to zero
            if (prefixSum == 0) {
                maxLength = i + 1;
            }

            // If the prefix sum has been seen before, calculate the length of the subarray
            if (prefixSumMap.containsKey(prefixSum)) {
                maxLength = Math.max(maxLength, i - prefixSumMap.get(prefixSum));
            } else {
                // Otherwise, store the current index for this prefix sum
                prefixSumMap.put(prefixSum, i);
            }
        }

        return maxLength;
    }
}



//* 9. Hashing - Sort Array in given Order

public class Solution {
    public int[] solve(int[] A, int[] B) {
        // Create a frequency map for elements in A
        HashMap<Integer, Integer> frequencyMap = new HashMap<>();
        for (int num : A) {
            frequencyMap.put(num, frequencyMap.getOrDefault(num, 0) + 1);
        }

        List<Integer> result = new ArrayList<>();

        // Add elements in the order specified by B
        for (int num : B) {
            if (frequencyMap.containsKey(num)) {
                int count = frequencyMap.get(num);
                while (count > 0) {
                    result.add(num);
                    count--;
                }
                frequencyMap.remove(num); // remove after processing
            }
        }

        // Collect the remaining elements not in B and sort them
        List<Integer> remaining = new ArrayList<>();
        for (Map.Entry<Integer, Integer> entry : frequencyMap.entrySet()) {
            int count = entry.getValue();
            while (count > 0) {
                remaining.add(entry.getKey());
                count--;
            }
        }
        Collections.sort(remaining);

        // Append sorted remaining elements to the result
        result.addAll(remaining);

        // Convert result list to array
        int[] resultArray = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            resultArray[i] = result.get(i);
        }

        return resultArray;
    }
}



//* 10. Hashing - Colorful Number

public class Solution {
    public int colorful(int A) {
        // Convert number to a string to easily access individual digits
        String numStr = Integer.toString(A);
        int length = numStr.length();

        // HashSet to store products
        HashSet<Integer> productSet = new HashSet<>();

        // Iterate through all subarrays
        for (int i = 0; i < length; i++) {
            int product = 1;
            for (int j = i; j < length; j++) {
                product *= (numStr.charAt(j) - '0');

                // If the product is already in the set, it's not a colorful number
                if (productSet.contains(product)) {
                    return 0;
                }

                // Add the product to the set
                productSet.add(product);
            }
        }

        // If all products are unique
        return 1;
    }
}




//* 11. Hashing - Count Subarrays

public class Solution {
    public int solve(int[] A) {
        int mod = 1000000007;
        HashMap<Integer, Integer> lastIndexMap = new HashMap<>();
        int n = A.length;
        int i = 0, result = 0;

        for (int j = 0; j < n; j++) {
            // If A[j] was seen before, move i to the right of the last seen index of A[j]
            if (lastIndexMap.containsKey(A[j])) {
                i = Math.max(i, lastIndexMap.get(A[j]) + 1);
            }
            // Calculate the number of subarrays ending at j with unique elements
            result = (result + (j - i + 1)) % mod;
            // Update the last seen index of A[j]
            lastIndexMap.put(A[j], j);
        }

        return result;
    }
}



//* 12. Hashing - Longest Substring Without Repeat


public class Solution {
    public int lengthOfLongestSubstring(String A) {
        int n = A.length();
        if (n == 0) return 0;

        HashMap<Character, Integer> lastIndexMap = new HashMap<>();
        int maxLength = 0;
        int i = 0; // Start of the current window

        for (int j = 0; j < n; j++) {
            char currentChar = A.charAt(j);

            // If the character is already in the map and within the current window
            if (lastIndexMap.containsKey(currentChar)) {
                i = Math.max(i, lastIndexMap.get(currentChar) + 1);
            }

            // Update the latest index of the character
            lastIndexMap.put(currentChar, j);

            // Calculate the current window length and update maxLength
            maxLength = Math.max(maxLength, j - i + 1);
        }

        return maxLength;
    }
}


//13. Length of longest consecutive ones

public class Solution {
    public int solve(String A) {
        int n = A.length();
        int totalOnes = 0;
        for(char c : A.toCharArray()){
            if(c == '1'){
                totalOnes++;
            }
        }

        if(totalOnes ==0){
            return 0;

        }
        int maxLength = 0;
        int currentLength = 0;
        int previousLength = -1;
        for(int i = 0; i < n; i++){
            if(A.charAt(i) == '1'){currentLength++;}
            else{
                if(previousLength != -1){
                    maxLength = Math.max(maxLength, previousLength + currentLength +1);
                }
                previousLength = currentLength;
                currentLength =0;
            }
            maxLength = Math.max(maxLength, currentLength +1);
        }
        maxLength = Math.max(maxLength, previousLength + currentLength +1);
        maxLength = Math.min(maxLength, totalOnes);
        return maxLength;
    }
}




//* 14. Majority Element

public class Solution {
    // DO NOT MODIFY THE ARGUMENTS WITH "final" PREFIX. IT IS READ ONLY
    public int majorityElement(final int[] A) {
        int candidate = -1;
        int count = 0;

        // Boyer-Moore Voting Algorithm
        for (int num : A) {
            if (count == 0) {
                candidate = num;
            }
            count += (num == candidate) ? 1 : -1;
        }

        // Given the problem constraints, we assume the candidate is the majority element
        return candidate;
    }
}



//* 15. N/3 Repeat Number

public class Solution {
    public int repeatedNumber(int[] A) {
        int n = A.length;

        // Step 1: Find potential candidates
        int candidate1 = -1, candidate2 = -1;
        int count1 = 0, count2 = 0;

        for (int num : A) {
            if (num == candidate1) {
                count1++;
            } else if (num == candidate2) {
                count2++;
            } else if (count1 == 0) {
                candidate1 = num;
                count1 = 1;
            } else if (count2 == 0) {
                candidate2 = num;
                count2 = 1;
            } else {
                count1--;
                count2--;
            }
        }

        // Step 2: Verify the candidates
        count1 = 0;
        count2 = 0;

        for (int num : A) {
            if (num == candidate1) {
                count1++;
            } else if (num == candidate2) {
                count2++;
            }
        }

        if (count1 > n / 3) {
            return candidate1;
        } else if (count2 > n / 3) {
            return candidate2;
        } else {
            return -1;
        }
    }
}



//* 16. Check Anagrams

public class Solution {
    public int solve(String A, String B) {
        // Check if lengths are different
        if (A.length() != B.length()) {
            return 0;
        }

        // Initialize frequency arrays for both strings
        int[] freqA = new int[26];
        int[] freqB = new int[26];

        // Count frequency of each character in A
        for (char c : A.toCharArray()) {
            freqA[c - 'a']++;
        }

        // Count frequency of each character in B
        for (char c : B.toCharArray()) {
            freqB[c - 'a']++;
        }

        // Compare frequency arrays
        for (int i = 0; i < 26; i++) {
            if (freqA[i] != freqB[i]) {
                return 0;
            }
        }

        return 1;
    }
}



//* 17. Colorful Number


public class Solution {
    public int colorful(int A) {
        // Convert number to string
        String numStr = Integer.toString(A);
        int n = numStr.length();
        Set<Integer> productSet = new HashSet<>();

        // Iterate over all possible starting points of substrings
        for (int i = 0; i < n; i++) {
            int product = 1;
            // Generate substrings of increasing lengths
            for (int j = i; j < n; j++) {
                // Calculate the product of the current substring
                product *= (numStr.charAt(j) - '0');

                // Check if this product is already in the set
                if (productSet.contains(product)) {
                    return 0; // Not colorful
                }

                // Add the product to the set
                productSet.add(product);
            }
        }

        return 1; // Colorful

    }
}
