/*
* 1. Linked List
* 2. Reverse Linked List
* 3. Palindrome List
* 4. Print Linked List
* 5. Insert in LikedList
* 6. Delete Linked List
* 7. Remove Duplicates from sorted List
* 8. Remove Nth Node from List End
* 9. K Reverse Linked List
* 10. Reverse Link List 2
* 11. Longest Palindromic List
* 12. Middle Element of Linked List
* 13. Merge Two Sorted Lists
* 14. Remove Loop from Linked List
* 15. Swap List Nodes in Pairs
* 16. Add Two Numbers as Lists
* 17. Reorder List
* 18. Copy List
* 19. LRU Cache
* 20. Partition List
* 21. Flatten a Linked List
* 22. Intersection of Linked List
* 23. Stacks - Passing Game
* 24. Stacks - Balanced Paranthesis
* 25. Stacks - Double Character Trouble
* 26. Stacks - Evaluate Expression
* 27. Stacks - Min Stack
* 28. Stacks - Infix to PostFix
* 29. Stacks - Redundant Braces
* 30.Stacks - Cjeck twobracket expressions
* 31. Stacks - Nearest Smaller Element
* 32. Stacks - Largest Rectangle in Histogram
* 33. Stacks - Max and Min
* 34. Stacks - Next Greater
* 35. Stacks - Maximum Rectangle
* 36. Stacks - Sort stack using another stack
* 37 . Queues - Queue Using Stacks
* 38. Queues - Parking Ice Cream Truck
* 39. Queues - Perfect Numbers
* 40. Queues - Reversing Elements of Queue
* 41. Queues - N integers containing only 1, 2 & 3
* 42. Queues - Sum of min and max
* 43. Queues - Unique Letter
*
 * */

//1. Linked List(Left)


//* 2. Reverse Linked List

public class Solution {
    public ListNode reverseList(ListNode A) {
        ListNode prev = null;
        ListNode current = A;
        ListNode next = null;
        while(current != null){
            next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }
        return prev;
    }
}

//* 3. Palindrome List

public class Solution {
    public int lPalin(ListNode head) {
        if (head == null || head.next == null) {
            return 1; // if List is Empty
        }

        ListNode slow = head;
        ListNode fast = head; // Slow and Fast Starting at Head

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next; // slow 1 step at a time fast 2 steps at a time
        }

        ListNode secondHalf = reverseList(slow); // from mid point (slow) list is reversed
// Compare Block
        ListNode firstHalf = head;
        ListNode secondHalfCopy = secondHalf;

        while (secondHalf != null) {
            if (firstHalf.val != secondHalf.val) {
                return 0;
            }
            firstHalf = firstHalf.next;
            secondHalf = secondHalf.next;
        }
//Compare Block
        reverseList(secondHalfCopy); // Re Reversing the list from midpoint(slow)
        // This is useful if the list needs to be preserved in its original form.

        return 1;
    }
    private ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode current = head;

        while (current != null) {
            ListNode next = current.next;
            current.next = prev;
            prev = current;
            current = next;
        }

        return prev;
    }
}


//* 4. Print Linked List

public class Solution {
    public void solve(ListNode A) {
        ListNode current = A;
        while(current != null){
            System.out.print(current.val + " ");
            current = current.next;
        }
        System.out.println();
    }
}
//* 5. Insert in LikedList

public class Solution {
    public ListNode solve(ListNode head, int B, int C) {
        ListNode newNode = new ListNode(B);
        if(C ==0){
            newNode.next = head;
            return newNode;
        }
        ListNode current = head;
        int currentPosition =0;
        while(current != null && currentPosition < C -1){
            current = current.next;
            currentPosition++;
        }
        if(current != null){
            newNode.next = current.next;
            current.next = newNode;
        }
        else{
            current = head;
            if(current == null){return newNode;}
            while (current.next != null){current = current.next;}
            current.next = newNode;
        }
        return head;
    }
}


//* 6. Delete Linked List

public class Solution {
    public ListNode solve(ListNode head, int B) {
        if(head == null){return null;}
        int length =0;
        ListNode current = head;
        while(current != null){length++; current = current.next;}
        if(B < 0 || B >= length){return head;}
        if(B ==0){return head.next;}
        current = head;
        int currentPosition = 0;
        while(current != null && currentPosition < B-1){current = current.next; currentPosition++;}
        if(current != null && current.next != null){current.next = current.next.next;}
        return head;
    }
}



//* 7. Remove Duplicates from sorted List

public class Solution {
    public ListNode deleteDuplicates(ListNode head) {
        ListNode current = head;
        while(current != null && current.next != null){
            if(current.val == current.next.val){
                current.next = current.next.next;
            }
            else{
                current = current.next;
            }
        }
        return head;
    }
}



//* 8. Remove Nth Node from List End


public class Solution {
    public ListNode removeNthFromEnd(ListNode head, int B) {
        ListNode fast = head;
        ListNode slow = head;
        for(int i = 0; i < B; i++){
            if(fast.next == null){return head.next;}
            fast = fast.next;
        }
        while (fast.next != null){
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return head;
    }
}



//* 9. K Reverse Linked List

public class Solution {
    public ListNode reverseList(ListNode head, int k) {
        if(head == null || k == 1){return head;}

        ListNode dumy = new ListNode(0);
        dumy.next = head;
        ListNode curr = dumy, nex = dumy, pre = dumy;
        int count = 0;
        while(curr.next != null){
            curr = curr.next;
            count++;
        }
        while (count >= k){
            curr = pre.next;
            nex = curr.next;
            for(int i = 1; i< k; i++){
                curr.next = nex.next;
                nex.next = pre.next;
                pre.next = nex;
                nex = curr.next;
            }
            pre = curr;
            count -= k;

        }
        return dumy.next;
    }
}




//* 10. Reverse Link List 2

public class Solution {
    public ListNode reverseBetween(ListNode head, int B, int C) {
        if(head == null || B == C){return head;}
        ListNode dumy = new ListNode(0);
        dumy.next = head;
        ListNode prev = dumy;
        for(int i = 1; i < B; i++){prev = prev.next;}

        ListNode start = prev.next;
        ListNode then = start.next;
        for(int i =0; i < C-B; i++){
            start.next = then.next;
            then.next = prev.next;
            prev.next = then;
            then = start.next;
        }
        return dumy.next;
    }
}



//* 11. Longest Palindromic List


public class Solution {
    public int solve(ListNode head) {
        if (head == null) return 0;

        // Step 1: Convert the linked list to an array
        ListNode current = head;
        int length = 0;
        while (current != null) {
            length++;
            current = current.next;
        }

        int[] values = new int[length];
        current = head;
        for (int i = 0; i < length; i++) {
            values[i] = current.val;
            current = current.next;
        }

        // Step 2: Find the longest palindromic subarray
        int maxLength = 1;
        for (int i = 0; i < length; i++) {
            // Odd length palindrome
            maxLength = Math.max(maxLength, expandAroundCenter(values, i, i));
            // Even length palindrome
            if (i + 1 < length) {
                maxLength = Math.max(maxLength, expandAroundCenter(values, i, i + 1));
            }
        }

        return maxLength;
    }

    // Helper function to expand around the center
    private int expandAroundCenter(int[] arr, int left, int right) {
        while (left >= 0 && right < arr.length && arr[left] == arr[right]) {
            left--;
            right++;
        }
        return right - left - 1;
    }
}

// Middle Element of Linked List

public class Solution {
    public int solve(ListNode head) {
        ListNode slowPointer = head;
        ListNode fastPointer = head;

        while(fastPointer != null && fastPointer.next != null){
            slowPointer = slowPointer.next;
            fastPointer = fastPointer.next.next;
        }
        return slowPointer.val;
    }
}


//* 13. Merge Two Sorted Lists

public class Solution {
    public ListNode mergeTwoLists(ListNode A, ListNode B) {
        ListNode dumy = new ListNode(0);
        ListNode tail = dumy;

        while(A != null && B != null){
            if(A.val <= B.val){
                tail.next = A;
                A = A.next;
            }
            else{
                tail.next = B;
                B = B.next;
            }
            tail = tail.next;
        }
        if(A!= null){tail.next = A;}
        if(B != null){tail.next = B;}
        return dumy.next;
    }
}



//* 14. Remove Loop from Linked List

public class Solution {
    public ListNode solve(ListNode head) {
        if (head == null) return head;

        // Step 1: Detect Cycle
        ListNode slow = head;
        ListNode fast = head;
        boolean hasCycle = false;

        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (slow == fast) {
                hasCycle = true;
                break;
            }
        }

        if (!hasCycle) {
            return head; // No cycle, return the original list
        }

        // Step 2: Find the start of the cycle
        ListNode cycleStart = head;
        while (cycleStart != slow) {
            cycleStart = cycleStart.next;
            slow = slow.next;
        }

        // Step 3: Break the cycle
        ListNode temp = cycleStart;
        while (temp.next != cycleStart) {
            temp = temp.next;
        }
        temp.next = null;

        return head;
    }
}





//* 15. Swap List Nodes in Pairs

public class Solution {
    public ListNode swapPairs(ListNode head) {
        ListNode dumy = new ListNode(0);
        dumy.next = head;
        ListNode current = dumy;

        while(current.next != null && current.next.next != null){
            ListNode first = current.next;
            ListNode second = current.next.next;

            first.next = second.next;
            second.next = first;
            current.next = second;
            current = first;
        }
        return dumy.next;
    }
}



//* 16. Add Two Numbers as Lists

public class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dumy = new ListNode(0);
        ListNode current = dumy;
        int carry = 0;
        while(l1 != null || l2 != null || carry != 0){
            int sum = carry;
            if(l1 != null){
                sum += l1.val;
                l1 = l1.next;

            }
            if(l2 != null){
                sum += l2.val;
                l2 = l2.next;
            }
            carry =  sum / 10;
            current.next = new ListNode(sum % 10);
            current = current.next;
        }
        return dumy.next;
    }
}



//* 17. Reorder List


public class Solution {
    public ListNode reorderList(ListNode head) {
        if(head == null || head.next == null){return head;}
        ListNode slow = head, fast = head;
        while(fast != null && fast.next != null){
            slow = slow.next;
            fast = fast.next.next;
        }
        ListNode prev = null, curr = slow, temp;
        while(curr != null){
            temp = curr.next;
            curr.next = prev;
            prev = curr;
            curr = temp;
        }
        ListNode first = head, second = prev;
        while(second.next != null){
            temp = first.next;
            first.next = second;
            first = temp;

            temp = second.next;
            second.next = first;
            second = temp;
        }
        return head;
    }
}


//18. Copy List

public class Solution {
    public RandomListNode copyRandomList(RandomListNode head) {
        if (head == null) return null;

        // Step 1: Clone nodes and insert them next to the original nodes
        RandomListNode current = head;
        while (current != null) {
            RandomListNode clone = new RandomListNode(current.label);
            clone.next = current.next;
            current.next = clone;
            current = clone.next;
        }

        // Step 2: Copy the random pointers for the cloned nodes
        current = head;
        while (current != null) {
            RandomListNode clone = current.next;
            clone.random = (current.random != null) ? current.random.next : null;
            current = clone.next;
        }

        // Step 3: Separate the original list and the cloned list
        RandomListNode newHead = head.next;
        RandomListNode cloneCurrent = newHead;
        current = head;

        while (current != null) {
            current.next = cloneCurrent.next;
            current = current.next;
            if (current != null) {
                cloneCurrent.next = current.next;
                cloneCurrent = cloneCurrent.next;
            }
        }

        return newHead;
    }
}


//* 19. LRU Cache


public class Solution {
    private class Node {
        int key, value;
        Node prev, next;
        Node(int key, int value) {
            this.key = key;
            this.value = value;
            this.prev = this.next = null;
        }
    }

    private final int capacity;
    private final HashMap<Integer, Node> map;
    private final Node head, tail;

    public Solution(int capacity) {
        this.capacity = capacity;
        this.map = new HashMap<>(capacity);
        this.head = new Node(-1, -1); // Dummy head
        this.tail = new Node(-1, -1); // Dummy tail
        head.next = tail;
        tail.prev = head;
    }
    private void remove(Node node) {
        Node prev = node.prev;
        Node next = node.next;
        prev.next = next;
        next.prev = prev;
    }

    private void add(Node node) {
        Node next = head.next;
        head.next = node;
        node.prev = head;
        node.next = next;
        next.prev = node;
    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) {
            return -1;
        }
        remove(node);
        add(node);
        return node.value;
    }

    public void set(int key, int value) {
        Node node = map.get(key);
        if (node != null) {
            node.value = value;
            remove(node);
            add(node);
        } else {
            if (map.size() == capacity) {
                Node lru = tail.prev;
                remove(lru);
                map.remove(lru.key);
            }
            Node newNode = new Node(key, value);
            add(newNode);
            map.put(key, newNode);
        }
    }
}



//* 20. Partition List

public class Solution {
    public ListNode partition(ListNode A, int B) {
        // Create dummy heads for the less and greater lists
        ListNode lessHead = new ListNode(0); // Dummy node for less than B
        ListNode greaterHead = new ListNode(0); // Dummy node for greater than or equal to B

        ListNode less = lessHead; // Pointer to build the less list
        ListNode greater = greaterHead; // Pointer to build the greater list

        // Traverse the original list
        ListNode current = A;
        while (current != null) {
            if (current.val < B) {
                less.next = current; // Add node to the less list
                less = less.next; // Move the less pointer
            } else {
                greater.next = current; // Add node to the greater list
                greater = greater.next; // Move the greater pointer
            }
            current = current.next; // Move to the next node
        }

        // Connect the end of the less list to the head of the greater list
        less.next = greaterHead.next;
        // End of the greater list should point to null
        greater.next = null;

        // Return the head of the new partitioned list
        return lessHead.next;
    }
}





//* 21. Flatten a Linked List

ListNode flatten(ListNode root) {
    if (root == null) return null;

    // Priority queue to maintain the order of nodes
    PriorityQueue<ListNode> minHeap = new PriorityQueue<>((a, b) -> Integer.compare(a.val, b.val));

    // Add the head of each sublist to the priority queue
    ListNode current = root;
    while (current != null) {
        minHeap.add(current);
        current = current.right;
    }

    // Dummy node to form the new flattened list
    ListNode dummy = new ListNode(0);
    ListNode tail = dummy;

    // Process the nodes in the priority queue
    while (!minHeap.isEmpty()) {
        ListNode node = minHeap.poll();
        tail.down = node; // Append the node to the flattened list
        tail = tail.down; // Move the tail pointer

        // If the node has a down pointer, add it to the queue
        if (node.down != null) {
            minHeap.add(node.down);
        }
    }

    return dummy.down;
}




//* 22. Intersection of Linked List


public class Solution {
    public ListNode getIntersectionNode(ListNode A, ListNode B) {

        // Step 1: Find lengths of both linked lists
        int lengthA = getLength(A);
        int lengthB = getLength(B);

        // Step 2: Align the start points of both lists
        if (lengthA > lengthB) {
            A = advanceByK(A, lengthA - lengthB);
        } else {
            B = advanceByK(B, lengthB - lengthA);
        }

        // Step 3: Traverse both lists to find the intersection
        while (A != null && B != null) {
            if (A == B) {
                return A; // Found intersection
            }
            A = A.next;
            B = B.next;
        }

        return null; // No intersection
    }

    // Helper function to find the length of a linked list
    private int getLength(ListNode head) {
        int length = 0;
        while (head != null) {
            length++;
            head = head.next;
        }
        return length;
    }

    // Helper function to advance the pointer by k nodes
    private ListNode advanceByK(ListNode head, int k) {
        while (k > 0 && head != null) {
            head = head.next;
            k--;
        }
        return head;
    }
}

//23. Stacks - Passing Game

public class Solution {
    public int solve(int A, int B, int[] C) {
        Stack<Integer> stack = new Stack<>();
        int currentPlayer = B;

        for(int i = 0; i< A; i++){
            if(C[i] == 0){
                if(!stack.isEmpty()){
                    currentPlayer = stack.pop();
                }
            }
            else{
                stack.push(currentPlayer);
                currentPlayer = C[i];
            }
        }
        return currentPlayer;
    }
}




//* 24. Stacks - Balanced Paranthesis

public class Solution {
    public int solve(String A) {
        Stack<Character> stack = new Stack<>();
        for(char ch : A.toCharArray()){
            if(ch == '(' || ch == '{' || ch == '['){
                stack.push(ch);
            }
            else if(ch == ')' || ch == '}' || ch == ']'){
                if(stack.isEmpty()){return 0;}

                char top = stack.pop();
                if((ch == ')' && top !='(') || (ch == '}' && top !='{') || (ch == ']' && top !='[')){
                    return 0;
                }
            }
        }
        return stack.isEmpty() ? 1:0;
    }
}




//* 25. Stacks - Double Character Trouble

public class Solution {
    public String solve(String A) {
        Stack<Character> stack = new Stack<>();

        for (char c : A.toCharArray()) {
            // If stack is not empty and top element is same as current character
            if (!stack.isEmpty() && stack.peek() == c) {
                stack.pop(); // Remove the top element (which is a pair with current character)
            } else {
                stack.push(c); // Otherwise, push the current character onto the stack
            }
        }

        // Build the final result from the stack
        StringBuilder result = new StringBuilder();
        while (!stack.isEmpty()) {
            result.insert(0, stack.pop()); // Insert each character at the beginning
        }

        return result.toString();
    }
}



//* 26. Stacks - Evaluate Expression

public class Solution {
    public int evalRPN(String[] A) {
        Stack<Integer> stack = new Stack<>();
        for(String token : A){
            switch(token){
                case "+":
                    stack.push(stack.pop() + stack.pop());
                    break;
                case "-":
                    int b = stack.pop();
                    int a = stack.pop();
                    stack.push(a - b);
                    break;
                case "*":
                    stack.push(stack.pop() * stack.pop());
                    break;
                case "/":
                    b = stack.pop();
                    a = stack.pop();
                    stack.push(a/b);
                    break;
                default:
                    stack.push(Integer.parseInt(token));
                    break;
            }
        }
        return stack.pop();
    }
}






//* 27. Stacks - Min Stack

import java.util.Stack;

public class Solution {
    private Stack<Integer> stack;
    private Stack<Integer> minStack;

    // Constructor to initialize the stacks
    public Solution() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }

    // Push element x onto stack
    public void push(int x) {
        stack.push(x);
        // If minStack is empty or x is smaller or equal to the current minimum, push x to minStack
        if (minStack.isEmpty() || x <= minStack.peek()) {
            minStack.push(x);
        }
    }

    // Removes the element on top of the stack
    public void pop() {
        if (stack.isEmpty()) return;
        int top = stack.pop();
        // If the popped element is the current minimum, pop it from minStack as well
        if (top == minStack.peek()) {
            minStack.pop();
        }
    }

    // Get the top element of the stack
    public int top() {
        if (stack.isEmpty()) return -1;
        return stack.peek();
    }

    // Retrieve the minimum element in the stack
    public int getMin() {
        if (minStack.isEmpty()) return -1;
        return minStack.peek();
    }
}



//* 28. Stacks - Infix to PostFix


public class Solution {
    public int precedence(char op) {
        switch (op) {
            case '^': return 3;
            case '*':
            case '/': return 2;
            case '+':
            case '-': return 1;
            default: return -1;
        }
    }
    public String solve(String A) {
        Stack<Character> stack = new Stack<>();
        StringBuilder result = new StringBuilder();

        // Helper function to determine operator precedence


        // Iterate through each character in the input string
        for (char c : A.toCharArray()) {
            if (Character.isLetter(c)) {
                // If the character is an operand, add it to the result
                result.append(c);
            } else if (c == '(') {
                // If the character is '(', push it to the stack
                stack.push(c);
            } else if (c == ')') {
                // If the character is ')', pop from the stack to the result
                // until '(' is encountered
                while (!stack.isEmpty() && stack.peek() != '(') {
                    result.append(stack.pop());
                }
                stack.pop(); // Pop the '(' from the stack
            } else {
                // If the character is an operator
                while (!stack.isEmpty() && precedence(stack.peek()) >= precedence(c)) {
                    result.append(stack.pop());
                }
                stack.push(c);
            }
        }

        // Pop all the remaining operators in the stack
        while (!stack.isEmpty()) {
            result.append(stack.pop());
        }

        return result.toString();
    }
}




//* 29. Stacks - Redundant Braces


public class Solution {
    public int braces(String A) {
        Stack<Character> stack = new Stack<>();
        for(char ch : A.toCharArray()){
            if(ch == ')'){
                char top = stack.pop();
                boolean hasOperator = false;
                while(top != '('){
                    if(top == '+' || top == '-' || top == '*' || top == '/'){
                        hasOperator = true;
                    }
                    top = stack.pop();
                }
                if(!hasOperator){
                    return 1;
                }
            }
            else{
                stack.push(ch);
            }
        }




        return 0;
    }
}




//* 30.Stacks - Cjeck twobracket expressions (Left)


//31. Stacks - Nearest Smaller Element


public class Solution {
    public int[] prevSmaller(int[] A) {
        int n = A.length;
        int[] G = new int[n];
        Stack<Integer> stack = new Stack<>();
        for(int i = 0; i < n; i++){
            while(!stack.isEmpty() && A[stack.peek()] >= A[i]){
                stack.pop();
            }
            if(stack.isEmpty()){
                G[i] = -1;
            }
            else{
                G[i] = A[stack.peek()];
            }
            stack.push(i);
        }
        return G;
    }
}



//* 32. Stacks - Largest Rectangle in Histogram

public class Solution {
    public int largestRectangleArea(int[] A) {
        int n = A.length;
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;

        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && A[stack.peek()] >= A[i]) {
                int height = A[stack.pop()];
                int width = stack.isEmpty() ? i : i - stack.peek() - 1;
                maxArea = Math.max(maxArea, height * width);
            }
            stack.push(i);
        }

        while (!stack.isEmpty()) {
            int height = A[stack.pop()];
            int width = stack.isEmpty() ? n : n - stack.peek() - 1;
            maxArea = Math.max(maxArea, height * width);
        }

        return maxArea;
    }

}





//* 33. Stacks - Max and Min

public class Solution {
    private static final int MOD = 1000000007;
    public int solve(int[] A) {
        int n = A.length;

        // Calculate the span where each element is the maximum
        long maxSum = calculateSpanContribution(A, true);

        // Calculate the span where each element is the minimum
        long minSum = calculateSpanContribution(A, false);

        // Result is the difference between maxSum and minSum
        long result = (maxSum - minSum + MOD) % MOD;
        return (int) result;
    }

    private long calculateSpanContribution(int[] A, boolean isMax) {
        int n = A.length;
        Stack<Integer> stack = new Stack<>();
        long totalSum = 0;

        // Arrays to store the next and previous less/greater element indices
        int[] next = new int[n];
        int[] prev = new int[n];

        // Calculate next and previous spans
        for (int i = 0; i < n; i++) {
            while (!stack.isEmpty() && (isMax ? A[stack.peek()] < A[i] : A[stack.peek()] > A[i])) {
                stack.pop();
            }
            prev[i] = stack.isEmpty() ? -1 : stack.peek();
            stack.push(i);
        }

        stack.clear();

        for (int i = n - 1; i >= 0; i--) {
            while (!stack.isEmpty() && (isMax ? A[stack.peek()] <= A[i] : A[stack.peek()] >= A[i])) {
                stack.pop();
            }
            next[i] = stack.isEmpty() ? n : stack.peek();
            stack.push(i);
        }

        // Calculate contributions
        for (int i = 0; i < n; i++) {
            long count = (long) (i - prev[i]) * (next[i] - i);
            totalSum = (totalSum + count * A[i]) % MOD;
        }

        return totalSum;
    }
}





//* 34. Stacks - Next Greater

public class Solution {
    public int[] nextGreater(int[] A) {
        int n = A.length;
        int[] result = new int[n];
        Stack<Integer> stack = new Stack<>();

        // Initialize all elements of result as -1
        for (int i = 0; i < n; i++) {
            result[i] = -1;
        }

        // Traverse the array
        for (int i = 0; i < n; i++) {
            // While stack is not empty and the current element is greater than
            // the element corresponding to the index on top of the stack
            while (!stack.isEmpty() && A[i] > A[stack.peek()]) {
                int index = stack.pop();
                result[index] = A[i];
            }
            // Push current index onto the stack
            stack.push(i);
        }

        // Remaining indices in stack have no greater element to their right
        // They are already set to -1

        return result;
    }
}



//* 35. Stacks - Maximum Rectangle

public class Solution {
    public int solve(int[][] A) {
        if (A == null || A.length == 0 || A[0].length == 0) {
            return 0;
        }

        int n = A.length;
        int m = A[0].length;
        int[] height = new int[m];
        int maxArea = 0;

        // Iterate through each row to build histograms
        for (int i = 0; i < n; i++) {
            // Update histogram heights
            for (int j = 0; j < m; j++) {
                if (A[i][j] == 0) {
                    height[j] = 0;
                } else {
                    height[j] += 1;
                }
            }

            // Calculate maximum area for the current histogram
            maxArea = Math.max(maxArea, largestRectangleArea(height));
        }

        return maxArea;
    }
    private int largestRectangleArea(int[] heights) {
        Stack<Integer> stack = new Stack<>();
        int maxArea = 0;
        int index = 0;

        while (index < heights.length) {
            if (stack.isEmpty() || heights[index] >= heights[stack.peek()]) {
                stack.push(index++);
            } else {
                int topOfStack = stack.pop();
                int area = heights[topOfStack] * (stack.isEmpty() ? index : index - stack.peek() - 1);
                maxArea = Math.max(maxArea, area);
            }
        }

        while (!stack.isEmpty()) {
            int topOfStack = stack.pop();
            int area = heights[topOfStack] * (stack.isEmpty() ? index : index - stack.peek() - 1);
            maxArea = Math.max(maxArea, area);
        }

        return maxArea;
    }
}




//* 36. Stacks - Sort stack using another stack


public class Solution {
    public int[] solve(int[] A) {
        Stack<Integer> originalStack = new Stack<>();
        Stack<Integer> sortedStack = new Stack<>();

        // Initialize originalStack with the input array
        for (int num : A) {
            originalStack.push(num);
        }

        // Sort the original stack using the sortedStack
        while (!originalStack.isEmpty()) {
            int temp = originalStack.pop();

            // Place the element in the correct position in sortedStack
            while (!sortedStack.isEmpty() && sortedStack.peek() > temp) {
                originalStack.push(sortedStack.pop());
            }

            sortedStack.push(temp);
        }

        // Transfer sortedStack to the result array in the correct order
        int[] result = new int[A.length];
        int index = 0;
        while (!sortedStack.isEmpty()) {
            result[index++] = sortedStack.pop();
        }

        // Reverse the result array to match the sorted stack
        reverse(result);

        return result;
    }

    // Helper method to reverse an array
    private void reverse(int[] arr) {
        int left = 0;
        int right = arr.length - 1;
        while (left < right) {
            int temp = arr[left];
            arr[left] = arr[right];
            arr[right] = temp;
            left++;
            right--;
        }
    }
}



//37 . Queues - Queue Using Stacks

public static class UserQueue {
    // Two stacks to maintain the queue
    private Stack<Integer> stack1;
    private Stack<Integer> stack2;

    /** Initialize your data structure here. */
    public UserQueue() {
        stack1 = new Stack<>();
        stack2 = new Stack<>();
    }

    /** Push element X to the back of queue. */
    public void push(int X) {
        stack1.push(X);
    }

    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        if (stack2.isEmpty()) {
            // Transfer all elements from stack1 to stack2
            while (!stack1.isEmpty()) {
                stack2.push(stack1.pop());
            }
        }
        // Now, stack2's top is the front of the queue
        return stack2.pop();
    }

    /** Get the front element of the queue. */
    public int peek() {
        if (stack2.isEmpty()) {
            // Transfer all elements from stack1 to stack2
            while (!stack1.isEmpty()) {
                stack2.push(stack1.pop());
            }
        }
        // Now, stack2's top is the front of the queue
        return stack2.peek();
    }

    /** Returns whether the queue is empty. */
    public boolean empty() {
        return stack1.isEmpty() && stack2.isEmpty();
    }
}

/**
 * Your UserQueue object will be instantiated and called as such:
 * UserQueue obj = new UserQueue();
 * obj.push(X);
 * int param2 = obj.pop();
 * int param3 = obj.peek();
 * boolean param4 = obj.empty();
 */



//* 38. Queues - Parking Ice Cream Truck
public class Solution {
    // DO NOT MODIFY THE ARGUMENTS WITH "final" PREFIX. IT IS READ ONLY
    public int[] slidingMaximum(final int[] A, int B) {
        int n = A.length;
        int[] result = new int[n - B + 1];
        if(B > n){
            int maxElement = Integer.MIN_VALUE;
            for(int num : A){
                if(num > maxElement){maxElement = num;}
            }
            return new int[]{maxElement};
        }
        Deque<Integer> deque = new ArrayDeque<>();

        for(int i = 0; i < B; i++){
            while(!deque.isEmpty() && A[deque.peekLast()] <= A[i]){deque.removeLast();}
            deque.addLast(i);
        }
        result[0] = A[deque.peekFirst()];

        for(int i = B; i< n; i++){
            while(!deque.isEmpty() && deque.peekFirst() <= i -B){
                deque.removeFirst();
            }
            while(!deque.isEmpty() && A[deque.peekLast()] <= A[i]){deque.removeLast();}
            deque.addLast(i);
            result[i - B + 1] = A[deque.peekFirst()];
        }
        return result;


    }
}





//* 39. Queues - Perfect Numbers


public class Solution {
    public String solve(int A) {
        Queue<String> queue = new LinkedList<>();
        queue.add("1");
        queue.add("2");

        String result = "";

        while (A > 0){
            String current = queue.poll();
            String palindrome = current + new StringBuilder(current).reverse().toString();
            A--;
            if(A == 0){
                result = palindrome;
                break;
            }
            queue.add(current + "1");
            queue.add(current + "2");
        }
        return result;
    }
}






//* 40. Queues - Reversing Elements of Queue
public class Solution {
    public int[] solve(int[] A, int B) {
        Queue<Integer> queue = new LinkedList<>();
        for(int i = 0; i< B; i++){
            queue.add(A[i]);
        }

        Stack<Integer> stack = new Stack<>();
        while(!queue.isEmpty()){
            stack.push(queue.remove());
        }
        int[] result = new int[A.length];
        int index = 0;
        while(!stack.isEmpty()){
            result[index++] = stack.pop();
        }
        for(int i = B; i < A.length; i++){
            result[index++] = A[i];
        }
        return result;
    }
}


//* 41. Queues - N integers containing only 1, 2 & 3

public class Solution {
    public int[] solve(int A) {
        List<Integer> result = new ArrayList<>();
        Queue<String> queue = new LinkedList<>();

        // Initial numbers
        queue.add("1");
        queue.add("2");
        queue.add("3");

        // Generate the first A numbers containing digits 1, 2, 3 only
        while (result.size() < A) {
            String current = queue.poll();
            result.add(Integer.parseInt(current));

            // Append '1', '2', '3' to the current number and add to the queue
            queue.add(current + "1");
            queue.add(current + "2");
            queue.add(current + "3");
        }

        // Convert the list to an array to return the result
        int[] resArray = new int[A];
        for (int i = 0; i < A; i++) {
            resArray[i] = result.get(i);
        }

        return resArray;
    }
}



//* 42. Queues - Sum of min and max(Left)



//* 43. Queues - Unique Letter


public class Solution {
    public String solve(String A) {
        // Initialize a Queue to keep track of characters in order
        Queue<Character> queue = new LinkedList<>();
        // Initialize a HashMap to keep track of character counts
        HashMap<Character, Integer> countMap = new HashMap<>();
        // StringBuilder to store the result
        StringBuilder result = new StringBuilder();

        // Process each character in the input string
        for (char c : A.toCharArray()) {
            // Update the count of the current character
            countMap.put(c, countMap.getOrDefault(c, 0) + 1);
            // Add the character to the queue
            queue.add(c);

            // Remove characters from the front of the queue that are no longer non-repeating
            while (!queue.isEmpty() && countMap.get(queue.peek()) > 1) {
                queue.poll();
            }

            // Append the first non-repeating character or '#' if all are repeating
            if (queue.isEmpty()) {
                result.append('#');
            } else {
                result.append(queue.peek());
            }
        }

        // Return the final result as a string
        return result.toString();
    }
}
