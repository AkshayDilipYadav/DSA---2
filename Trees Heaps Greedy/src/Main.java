/*
* 1. Preorder Traversal
* 2. Inorder Traversal
* 3. Binary Tree from Inorder and Postorder
* 4. postorder Traversal
* 5. Binary Tree from Inorder and Preorder
* 6. Level Order
* 7. Right View of Binary Tree
* 8. Vertical Order Traversal
* 9. Balanced Binary Tree
* 10. ZigZag Level order Traversal BT
* 11. Serialize Binary Tree
* 12. Deserialize Binary Tree
* 13. Top View of Binary Tree
* 14. Search in BST
* 15. Delete a node in BST
* 16. Sorted Array to Balanced BST
* 17. Valid Binary Search Tree
* 18. Check for BST with one Child
* 19. BST nodes in a range
* 20. Two Sum BST
* 21. Kth Smallest Element in BST
* 22. Least Common Ancestor
* 23. LCA in BST
* 24. Node to Root path in Binary Tree
* 25. Common Nodes in Two BST
* 26. Distance between Nodes of BST
* 27. Morris Inorder Traversal
*28. Invert the Binary Tree
* 29. Equal Tree Partition
* 30. Diameter of Binary Tree
* 31. Identical Binary Tree
* 32. Symmetric BInary Tree
* 33. Heaps - Ath Largest Element
* 34.Heaps - K places Apart
* 35. Heaps - Running Median
* 36. Heaps - Bth Smallest Element
* 36. Heaps - Heap Queries
* 37. Heaps - Build a Heap
* 38. Heaps - Maximum array sum after B negations
* 39. Heaps - Misha and Candies
* 40. Heaps - Minimum largest element
* 41. Heaps - Merge K sorted Lists
* 42. Heaps - Build a Heap
* 43. Heaps - Ways to form Max Heap
* 44. Heaps - Product of 3
* 45. Heaps - Kth Smallest Element in a sorted matrix
* 46. Greedy - Flipkart's Challenge in Effective Inventory Management
* 47. Greedy - Finish Maximum Jobs
* 48. Greedy - Distribute Candy
* 49. Greedy - Another COin Problem
* 50. Greedy - Seats
* 51. Greedy - Assign Mice to Holes
 * */

//1. Preorder Traversal

public class Solution {
    public int[] preorderTraversal(TreeNode A) {
        List<Integer> result = new ArrayList<>();
        preorder(A, result);
        int[] resArray = new int[result.size()];
        for(int i = 0; i < result.size(); i++){
            resArray[i] = result.get(i);
        }
        return resArray;
    }
    public void preorder(TreeNode node, List<Integer> result){
        if(node == null){return;}
        result.add(node.val);
        preorder(node.left,result);
        preorder(node.right, result);

    }
}

//* 2. Inorder Traversal


public class Solution {
    public int[] inorderTraversal(TreeNode A) {
        List<Integer> result = new ArrayList<>();
        inorder(A, result);
        int[] resArray = new int[result.size()];
        for(int i = 0; i < result.size(); i++){
            resArray[i] = result.get(i);
        }
        return resArray;
    }
    public void inorder(TreeNode node, List<Integer> result){
        if(node == null){return;}
        inorder(node.left,result);
        result.add(node.val);

        inorder(node.right, result);

    }
}




//* 3. Binary Tree from Inorder and Postorder

public class Solution {
    private int postIndex;
    public TreeNode buildTree(int[] inorder, int[] postorder){
        postIndex = postorder.length - 1;
        return buildTreeHelper(inorder, postorder, 0, inorder.length - 1);
    }
    private TreeNode buildTreeHelper(int[] inorder, int[] postorder, int inStart, int inEnd){
        if(inStart > inEnd){
            return null;
        }
        int rootVal = postorder[postIndex--];
        TreeNode root = new TreeNode(rootVal);
        int inIndex = findInIndex(inorder, inStart, inEnd, rootVal);
        root.right = buildTreeHelper(inorder, postorder, inIndex + 1, inEnd);
        root.left = buildTreeHelper(inorder, postorder, inStart, inIndex - 1);
        return root;
    }
    private int findInIndex(int[] inorder, int start, int end, int value){
        for(int i = start; i <= end; i++){
            if(inorder[i] == value){return i;}
        }
        return -1;
    }
}


//* 4. postorder Traversal

public class Solution {
    public int[] postorderTraversal(TreeNode A) {
        List<Integer> result = new ArrayList<>();
        postorder(A, result);
        int[] resArray = new int[result.size()];
        for(int i = 0; i < result.size(); i++){
            resArray[i] = result.get(i);
        }
        return resArray;
    }
    private void postorder(TreeNode node, List<Integer> result){
        if(node == null){return;}
        postorder(node.left, result);
        postorder(node.right, result);
        result.add(node.val);
    }
}


//* 5. Binary Tree from Inorder and Preorder


public class Solution {
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        return buildTreeHelper(preorder, inorder, 0, 0, inorder.length -1);
    }
    private TreeNode buildTreeHelper(int[] preorder, int[] inorder, int preStart, int inStart, int inEnd){
        if(preStart > preorder.length - 1 || inStart > inEnd){return null;}
        TreeNode root = new TreeNode(preorder[preStart]);
        int inIndex = 0;
        for(int i = inStart; i <= inEnd; i++){
            if(inorder[i] == root.val){
                inIndex = i;
            }
        }
        root.left = buildTreeHelper(preorder, inorder, preStart + 1, inStart, inIndex -1);
        root.right = buildTreeHelper(preorder, inorder, preStart + inIndex - inStart + 1, inIndex + 1, inEnd);
        return root;
    }
}

//6. Level Order

public class Solution {
    public int[][] solve(TreeNode root) {
        if(root == null){return new int[0][0];}
        int height = getHeight(root);
        int[][] result = new int[height][];
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        for(int level = 0; level < height; level++){
            int levelSize = queue.size();
            result[level] = new int[levelSize];
            for(int i = 0; i < levelSize; i++){
                TreeNode currentNode = queue.poll();
                result[level][i] = currentNode.val;
                if(currentNode.left != null){queue.add(currentNode.left);}
                if(currentNode.right != null){queue.add(currentNode.right);}
            }
        }
        return result;
    }

    private int getHeight(TreeNode root){
        if(root == null){return 0;}
        int leftHeight = getHeight(root.left);
        int rightHeight = getHeight(root.right);
        return Math.max(leftHeight, rightHeight)+ 1;
    }
}


//* 7. Right View of Binary Tree

public class Solution {
    public static int getHeight (TreeNode root){
        if(root == null){return 0;}
        int leftHeight = getHeight(root.left);
        int rightHeight = getHeight(root.right);
        return Math.max(leftHeight, rightHeight) + 1;
    }
    public int[] solve(TreeNode root) {
        if(root == null){return new int[0];}
        int height = getHeight(root);
        int[] rightViewArray = new int[height];
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        int level = 0;
        while(!queue.isEmpty()){
            int levelSize = queue.size();
            TreeNode rightMostNode = null;
            for(int i = 0; i< levelSize; i++){
                TreeNode node = queue.poll();
                rightMostNode = node;
                if(node.left != null){queue.add(node.left);}
                if(node.right != null){queue.add(node.right);}
            }
            if(rightMostNode != null){
                rightViewArray[level] = rightMostNode.val;
            }
            level++;
        }
        return rightViewArray;
    }
}



//* 8. Vertical Order Traversal

public class Solution {
    public int[][] verticalOrderTraversal(TreeNode root) {
        if(root == null){return new int[0][0];}
        TreeMap<Integer, List<Integer>> map = new TreeMap<>();
        Queue<TreeNode> nodeQueue = new LinkedList<>();
        Queue<Integer> levelQueue = new LinkedList<>();
        nodeQueue.add(root);
        levelQueue.add(0);
        while(!nodeQueue.isEmpty()){
            TreeNode node = nodeQueue.poll();
            int level = levelQueue.poll();
            if(!map.containsKey(level)){
                map.put(level, new LinkedList<>());
            }
            map.get(level).add(node.val);
            if(node.left != null){
                nodeQueue.add(node.left);
                levelQueue.add(level - 1);
            }
            if(node.right != null){
                nodeQueue.add(node.right);
                levelQueue.add(level + 1);
            }
        }
        int[][] result = new int[map.size()][];
        int i = 0;
        for(List<Integer> list : map.values()){
            result[i] = new int[list.size()];
            int j = 0;
            for(int val : list){
                result[i][j++] = val;
            }
            i++;
        }
        return result;
    }
}



//* 9. Balanced Binary Tree

public class Solution {
    public int isBalanced(TreeNode root){
        return isBalancedHelper(root) != -1 ? 1:0;
    }
    public int isBalancedHelper(TreeNode node) {
        if(node == null){return 0;}
        int leftHeight = isBalancedHelper(node.left);
        if(leftHeight == -1){return -1;}
        int rightHeight = isBalancedHelper(node.right);
        if(rightHeight == -1){return -1;}
        if(Math.abs(leftHeight - rightHeight)> 1){return -1;}
        return Math.max(leftHeight, rightHeight)+ 1;
    }
}
//* 10. ZigZag Level order Traversal BT

public class Solution {
    public int[][] zigzagLevelOrder(TreeNode A) {
        if (A == null) return new int[0][0];

        List<List<Integer>> result = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(A);
        boolean leftToRight = true;

        while (!queue.isEmpty()) {
            int levelSize = queue.size();
            List<Integer> level = new ArrayList<>(levelSize);

            for (int i = 0; i < levelSize; i++) {
                TreeNode node = queue.poll();
                if (leftToRight) {
                    level.add(node.val);
                } else {
                    level.add(0, node.val); // Add to the beginning for reverse order
                }

                if (node.left != null) queue.add(node.left);
                if (node.right != null) queue.add(node.right);
            }

            result.add(level);
            leftToRight = !leftToRight; // Toggle the order
        }

        // Convert result list of lists to 2D array
        int[][] finalResult = new int[result.size()][];
        for (int i = 0; i < result.size(); i++) {
            finalResult[i] = result.get(i).stream().mapToInt(Integer::intValue).toArray();
        }

        return finalResult;
    }
}


//* 11. Serialize Binary Tree

public class Solution {
    public int[] solve(TreeNode A) {
        if (A == null) {
            return new int[]{};
        }

        List<Integer> result = new ArrayList<>();
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(A);

        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();

            if (node != null) {
                result.add(node.val);
                queue.add(node.left);  // Add left child to queue (can be null)
                queue.add(node.right); // Add right child to queue (can be null)
            } else {
                result.add(-1); // Represents a null child
            }
        }

        // Convert result list to array
        int[] serializedTree = new int[result.size()];
        for (int i = 0; i < result.size(); i++) {
            serializedTree[i] = result.get(i);
        }

        return serializedTree;
    }
}


//* 12. Deserialize Binary Tree

public class Solution {
    public TreeNode solve(int[] A) {
        if (A == null || A.length == 0) {
            return null;
        }

        // Initialize the root of the tree
        TreeNode root = new TreeNode(A[0]);
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);

        int i = 1;
        while (i < A.length) {
            // Dequeue the front node
            TreeNode currentNode = queue.poll();

            // Process the left child
            if (A[i] != -1) {
                currentNode.left = new TreeNode(A[i]);
                queue.add(currentNode.left);
            }
            i++;

            // Ensure there's a right child to process
            if (i < A.length && A[i] != -1) {
                currentNode.right = new TreeNode(A[i]);
                queue.add(currentNode.right);
            }
            i++;
        }

        return root;
    }
}


//* 13. Top View of Binary Tree


public class Solution {

    class QueueNode{
        TreeNode node;
        int hd;
        QueueNode(TreeNode n, int h){
            node = n;
            hd = h;
        }
    }

    public int[] solve(TreeNode root) {
        if(root == null){return new int[0];}
        Map<Integer, Integer> map = new TreeMap<>();
        Queue<QueueNode> queue = new LinkedList<>();
        queue.add(new QueueNode(root, 0));

        while(!queue.isEmpty()){
            QueueNode temp = queue.poll();
            int hd = temp.hd;
            TreeNode node = temp.node;
            if(!map.containsKey(hd)){
                map.put(hd, node.val);
            }
            if(node.left != null){
                queue.add(new QueueNode(node.left, hd - 1));
            }
            if(node.right != null){
                queue.add(new QueueNode(node.right, hd + 1));
            }
        }
        int[] topView = new int[map.size()];
        int index = 0;
        for(Map.Entry<Integer, Integer> entry : map.entrySet()){
            topView[index++] = entry.getValue();
        }
        return topView;
    }
}


//14. Search in BST

public class Solution {
    public int solve(TreeNode root, int B) {
        return search(root, B) ? 1: 0;
    }
    private boolean search(TreeNode root, int val){
        if(root == null){return false;}
        if(root.val == val){return true;}
        if(val < root.val){return search(root.left, val);}
        return search(root.right, val);
    }
}



//* 15. Delete a node in BST

public class Solution {
    public TreeNode solve(TreeNode root, int B) {
        if(root == null){
            return null;
        }
        if(B < root.val){root.left = solve(root.left, B);}
        else if(B > root.val){root.right = solve(root.right, B);}
        else{
            if(root.left == null){return root.right;}
            else if(root.right == null){return root.left;}
            else{TreeNode predecessor = findMax(root.left);
                root.val = predecessor.val;
                root.left = solve(root.left, predecessor.val);}
        }
        return root;
    }
    private TreeNode findMax(TreeNode node){
        while(node.right != null){
            node = node.right;
        }
        return node;
    }
}



//* 16. Sorted Array to Balanced BST


public class Solution {
    // DO NOT MODIFY THE ARGUMENTS WITH "final" PREFIX. IT IS READ ONLY
    public TreeNode sortedArrayToBST(final int[] A) {
        if(A == null || A.length == 0){
            return null;
        }
        return sortedArrayToBSTHelper(A, 0, A.length -1);
    }
    private TreeNode sortedArrayToBSTHelper(int[] A, int start, int end){
        if(start > end){return null;}
        int mid = start + (end - start)/2;
        TreeNode root = new TreeNode(A[mid]);
        root.left = sortedArrayToBSTHelper(A, start, mid -1);
        root.right = sortedArrayToBSTHelper(A, mid + 1, end);
        return root;
    }

}


//* 17. Valid Binary Search Tree

public class Solution {
    private TreeNode prev;
    public int isValidBST(TreeNode root) {
        prev = null;
        return isBST(root) ? 1:0;
    }
    private boolean isBST(TreeNode node){
        if(node == null){return true;}
        if(!isBST(node.left)){return false;}
        if(prev != null && node.val <= prev.val){return false;}
        prev = node;
        return isBST(node.right);
    }
}


//* 18. Check for BST with one Child (Left)
//* 19. BST nodes in a range

public class Solution {
    public int solve(TreeNode A, int B, int C) {
        return countNodesInRange(A, B, C);
    }

    private int countNodesInRange(TreeNode node, int B, int C) {
        // Base case: if the node is null, return 0
        if (node == null) {
            return 0;
        }

        // Initialize count to 0
        int count = 0;

        // Check if the current node's value is within the range [B, C]
        if (node.val >= B && node.val <= C) {
            count = 1;
        }

        // Recurse to the left subtree if the current node's value is greater than B
        if (node.val > B) {
            count += countNodesInRange(node.left, B, C);
        }

        // Recurse to the right subtree if the current node's value is less than C
        if (node.val < C) {
            count += countNodesInRange(node.right, B, C);
        }

        return count;
    }
}



//* 20. Two Sum BST


public class Solution {
    public int t2Sum(TreeNode A, int B) {
        Set<Integer> seen = new HashSet<>();
        return findPairWithSum(A, B, seen) ? 1 : 0;
    }

    private boolean findPairWithSum(TreeNode node, int target, Set<Integer> seen) {
        if (node == null) {
            return false;
        }

        // Check if the complement of the current node's value exists in the set
        if (seen.contains(target - node.val)) {
            return true;
        }

        // Add the current node's value to the set
        seen.add(node.val);

        // Recursively check left and right subtrees
        return findPairWithSum(node.left, target, seen) || findPairWithSum(node.right, target, seen);
    }
}



// 21. Kth Smallest Element in BST

public class Solution {
    public int kthsmallest(TreeNode root, int B) {
        return inorderTraversal(root, B);
    }
    private int inorderTraversal(TreeNode root, int k){
        Stack<TreeNode> stack = new Stack<>();
        TreeNode current = root;
        int count = 0;
        while(current != null || !stack.isEmpty()){
            while(current != null){
                stack.push(current);
                current = current.left;
            }
            current = stack.pop();
            count++;
            if(count == k){
                return current.val;
            }
            current = current.right;
        }
        throw new IllegalArgumentException("Tree has fewer than B nodes");
    }
}

//* 22. Least Common Ancestor


public class Solution {
    public int lca(TreeNode A, int B, int C) {
        if (A == null) return -1;

        // If B and C are the same, we should search the tree and return the node if found
        if (B == C) {
            boolean exists = nodeExists(A, B);
            return exists ? B : -1;
        }

        TreeNode lcaNode = findLCA(A, B, C);

        // If LCA exists, we should ensure both B and C are actually present in the tree
        if (lcaNode != null && nodeExists(A, B) && nodeExists(A, C)) {
            return lcaNode.val;
        }

        return -1;
    }

    private TreeNode findLCA(TreeNode root, int B, int C) {
        // Base case
        if (root == null) return null;
        if (root.val == B || root.val == C) return root;

        // Search in left and right subtrees
        TreeNode leftLCA = findLCA(root.left, B, C);
        TreeNode rightLCA = findLCA(root.right, B, C);

        // If both left and right subtrees return non-null, this is the LCA
        if (leftLCA != null && rightLCA != null) return root;

        // Otherwise, return the non-null child (if both are null, it returns null)
        return leftLCA != null ? leftLCA : rightLCA;
    }

    // Utility function to check if a node with the given value exists in the tree
    private boolean nodeExists(TreeNode root, int val) {
        if (root == null) return false;
        if (root.val == val) return true;
        return nodeExists(root.left, val) || nodeExists(root.right, val);
    }
}


//* 23. LCA in BST

public class Solution {
    public int solve(TreeNode A, int B, int C) {
        return findLCA(A, B, C).val;
    }

    private TreeNode findLCA(TreeNode root, int B, int C) {
        // Start from the root node
        while (root != null) {
            // If both B and C are smaller than root, go to the left subtree
            if (B < root.val && C < root.val) {
                root = root.left;
            }
            // If both B and C are greater than root, go to the right subtree
            else if (B > root.val && C > root.val) {
                root = root.right;
            }
            // Else the current node is the LCA
            else {
                return root;
            }
        }
        return null; // This will never be reached as per problem guarantees
    }
}


//* 24. Node to Root path in Binary Tree

public class Solution {
    public int[] getPathToRoot(TreeNode A, int B) {
        List<Integer> path = new ArrayList<>();
        findPath(A, B, path);
        // Reverse the path to get the correct order from B to root
        Collections.reverse(path);
        // Convert the list to an array before returning
        int[] result = new int[path.size()];
        for (int i = 0; i < path.size(); i++) {
            result[i] = path.get(i);
        }
        return result;
    }

    private boolean findPath(TreeNode root, int B, List<Integer> path) {
        if (root == null) return false;

        // Add current node to the path
        path.add(root.val);

        // If current node is the target node B
        if (root.val == B) {
            return true;
        }

        // Recursively search in the left or right subtree
        if (findPath(root.left, B, path) || findPath(root.right, B, path)) {
            return true;
        }

        // If not found in either subtree, remove current node from the path
        path.remove(path.size() - 1);
        return false;
    }
}


//* 25. Common Nodes in Two BST

public class Solution {
    private static final int MOD = 1000000007;

    public int solve(TreeNode A, TreeNode B) {
        long sum = 0;

        Stack<TreeNode> stackA = new Stack<>();
        Stack<TreeNode> stackB = new Stack<>();

        TreeNode currA = A;
        TreeNode currB = B;

        while ((currA != null || !stackA.isEmpty()) && (currB != null || !stackB.isEmpty())) {
            // Traverse the left subtree of A and B
            while (currA != null) {
                stackA.push(currA);
                currA = currA.left;
            }

            while (currB != null) {
                stackB.push(currB);
                currB = currB.left;
            }

            currA = stackA.peek();
            currB = stackB.peek();

            if (currA.val == currB.val) {
                // Common node found
                sum = (sum + currA.val) % MOD;
                stackA.pop();
                stackB.pop();
                currA = currA.right;
                currB = currB.right;
            } else if (currA.val < currB.val) {
                // A's node is smaller, move forward in A
                stackA.pop();
                currA = currA.right;
                currB = null; // Hold B in place
            } else {
                // B's node is smaller, move forward in B
                stackB.pop();
                currB = currB.right;
                currA = null; // Hold A in place
            }
        }

        return (int) sum;
    }
}


//* 26. Distance between Nodes of BST


public class Solution {
    public int solve(TreeNode A, int B, int C) {
        TreeNode lca = findLCA(A, B, C);
        int distB = findDistance(lca, B);
        int distC = findDistance(lca, C);
        return distB + distC;
    }

    // Function to find the Lowest Common Ancestor (LCA)
    private TreeNode findLCA(TreeNode root, int B, int C) {
        if (root == null) return null;

        if (root.val > B && root.val > C) {
            return findLCA(root.left, B, C);
        } else if (root.val < B && root.val < C) {
            return findLCA(root.right, B, C);
        } else {
            return root;
        }
    }

    // Function to find the distance from a given node to the target
    private int findDistance(TreeNode root, int target) {
        if (root == null) return 0;

        if (root.val == target) {
            return 0;
        } else if (root.val > target) {
            return 1 + findDistance(root.left, target);
        } else {
            return 1 + findDistance(root.right, target);
        }
    }
}

//27. Morris Inorder Traversal

public class Solution {
    public ArrayList<Integer> solve(TreeNode root) {
        ArrayList<Integer> result = new ArrayList<>();
        TreeNode current = root;
        while(current != null){
            if(current.left == null){
                result.add(current.val);
                current = current.right;
            }
            else{
                TreeNode predecessor = current.left;
                while(predecessor.right != null && predecessor.right != current){
                    predecessor = predecessor.right;
                }
                if(predecessor.right == null){
                    predecessor.right = current;
                    current = current.left;
                }
                else{
                    predecessor.right = null;
                    result.add(current.val);
                    current = current.right;
                }
            }
        }
        return result;
    }
}
//*28. Invert the Binary Tree

public class Solution {
    public TreeNode invertTree(TreeNode root) {
        if(root == null){
            return null;
        }
        TreeNode left = invertTree(root.left);
        TreeNode right = invertTree(root.right);
        root.left = right;
        root.right = left;
        return root;
    }
}


//* 29. Equal Tree Partition


public class Solution {
    private long totalSum = 0; // Use long to handle large sums
    private boolean canPartition = false;

    public int solve(TreeNode A) {
        // Step 1: Calculate the total sum of the tree
        totalSum = calculateSum(A);

        // If the total sum is odd, it's not possible to partition into equal halves
        if (totalSum % 2 != 0) {
            return 0;
        }

        // Step 2: Check if there's a subtree with sum equal to half of the total sum
        checkPartition(A, totalSum / 2);

        return canPartition ? 1 : 0;
    }

    // Function to calculate the sum of the tree
    private long calculateSum(TreeNode node) {
        if (node == null) return 0;

        return node.val + calculateSum(node.left) + calculateSum(node.right);
    }

    // Function to check if there is a subtree with the given target sum
    private long checkPartition(TreeNode node, long targetSum) {
        if (node == null) return 0;

        long currentSum = node.val + checkPartition(node.left, targetSum) + checkPartition(node.right, targetSum);

        // If we find a subtree with the sum equal to targetSum and it's not the entire tree
        if (currentSum == targetSum && currentSum != totalSum) {
            canPartition = true;
        }

        return currentSum;
    }
}



//* 30. Diameter of Binary Tree

public class Solution {
    private int maxDiameter = 0;
    public int solve(TreeNode root) {
        calculateHeight(root);
        return maxDiameter;
    }
    private int calculateHeight(TreeNode node){
        if(node == null){return 0;}
        int leftHeight = calculateHeight(node.left);
        int rightHeight = calculateHeight(node.right);
        int currentDiameter = leftHeight + rightHeight;
        maxDiameter = Math.max(maxDiameter, currentDiameter);
        return Math.max(leftHeight, rightHeight)+ 1;
    }
}



//* 31. Identical Binary Tree

public class Solution {
    public int isSameTree(TreeNode A, TreeNode B) {
        return isIdentical(A, B) ? 1 : 0;
    }

    // Helper function to check if two trees are identical
    private boolean isIdentical(TreeNode a, TreeNode b) {
        if (a == null && b == null) {
            return true;
        }
        if (a == null || b == null) {
            return false;
        }
        if (a.val != b.val) {
            return false;
        }
        // Recursively check left and right subtrees
        return isIdentical(a.left, b.left) && isIdentical(a.right, b.right);
    }
}


//* 32. Symmetric BInary Tree

public class Solution {
    public int isSymmetric(TreeNode A) {
        return isMirror(A, A) ? 1 : 0;
    }

    // Helper function to check if two trees are mirror images of each other
    private boolean isMirror(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) {
            return true;
        }
        if (t1 == null || t2 == null) {
            return false;
        }
        return (t1.val == t2.val)
                && isMirror(t1.left, t2.right)
                && isMirror(t1.right, t2.left);
    }
}

//33. Heaps - Ath Largest Element

public class Solution {
    public int[] solve(int A, int[] B) {
        int N = B.length;
        int[] result = new int[N];

        PriorityQueue<Integer> minHeap = new PriorityQueue<>(A);
// Priority queue size of A value

        for(int i = 0 ; i < N; i++){

            if(minHeap.size() < A){minHeap.add(B[i]);}
            else if(B[i] > minHeap.peek()){
                minHeap.poll();
                minHeap.add(B[i]);
            }
            if(minHeap.size() < A){result[i] = -1;}
            else{result[i] = minHeap.peek();}
        }
        return result;
    }
}



//* 34.Heaps - K places Apart

public class Solution {
    public int[] solve(int[] A, int B) {
        int N = A.length;
        int[] result = new int[N];
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for(int i = 0; i <=B && i < N; i++){ // till B + 1
            minHeap.add(A[i]);
        }
        int index = 0;

        for(int i = B + 1; i < N; i++){ // from B + 1
            result[index++] = minHeap.poll();
            minHeap.add(A[i]);
        }

        while(!minHeap.isEmpty()){
            result[index++] = minHeap.poll();
        }
        return result;
    }
}




//* 35. Heaps - Running Median

public class Solution {
    public int[] solve(int[] A) {
        int N = A.length;
        int[] C = new int[N];
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for(int i = 0; i < N; i++){
            maxHeap.add(A[i]);
            minHeap.add(maxHeap.poll());
            if(maxHeap.size() < minHeap.size()){
                maxHeap.add(minHeap.poll());
            }
            C[i] = maxHeap.peek();
        }
        return C;
    }
}




//* 36. Heaps - Bth Smallest Element

public class Solution {
    public int bthSmallest(int[] A, int B) {
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Collections.reverseOrder());
        for(int i = 0; i < B ; i ++){
            maxHeap.add(A[i]);
        }
        for(int i = B ; i < A.length; i++){
            if(A[i] < maxHeap.peek()){
                maxHeap.poll();
                maxHeap.add(A[i]);
            }

        }
        return maxHeap.peek();
    }
}





//* 36. Heaps - Heap Queries

public class Solution {
    public int[] solve(int[][] A) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        List<Integer> results = new ArrayList<>();

        for (int[] query : A) {
            int P = query[0];
            int Q = query[1];

            if (P == 1) {
                if (minHeap.isEmpty()) {
                    results.add(-1);
                } else {
                    results.add(minHeap.poll());
                }
            } else if (P == 2) {
                minHeap.add(Q);
            }
        }

        int[] resultArray = new int[results.size()];
        for (int i = 0; i < results.size(); i++) {
            resultArray[i] = results.get(i);
        }

        return resultArray;
    }
}





//* 37. Heaps - Build a Heap


class Solution {
    public int[] buildHeap(int[] A) {
        int n = A.length;

        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(A, n, i);
        }

        return A;
    }
    public static void heapify(int[] array, int n, int i) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && array[left] < array[smallest]) {
            smallest = left;
        }

        if (right < n && array[right] < array[smallest]) {
            smallest = right;
        }

        if (smallest != i) {
            int swap = array[i];
            array[i] = array[smallest];
            array[smallest] = swap;

            heapify(array, n, smallest);
        }
    }
}



//* 38. Heaps - Maximum array sum after B negations

public class Solution {
    public int solve(int[] A, int B) {
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();
        for (int num : A) {
            minHeap.add(num);
        }

        // Step 2: Perform B modifications
        while (B-- > 0) {
            // Extract the smallest element
            int minElement = minHeap.poll();
            // Flip its sign
            minElement = -minElement;
            // Add it back to the heap
            minHeap.add(minElement);
        }

        // Step 3: Calculate the final sum
        int sum = 0;
        while (!minHeap.isEmpty()) {
            sum += minHeap.poll();
        }

        return sum;
    }
}




//* 39. Heaps - Misha and Candies (Left)
//* 40. Heaps - Minimum largest element (Left)



//41. Heaps - Merge K sorted Lists

public class Solution {
    public ListNode mergeKLists(ArrayList<ListNode> a) {
        PriorityQueue<ListNode> minHeap = new PriorityQueue<>(Comparator.comparingInt(node -> node.val));
        /* Initialization-
        this priority queue efficiently retrieve the smallest element from the heap
         */
        for(ListNode list : a){
            if(list != null){minHeap.add(list);}
        }
        ListNode dummy = new ListNode(0);
        ListNode current = dummy;
        while(!minHeap.isEmpty()){
            ListNode node = minHeap.poll();
            current.next = node;
            current = current.next;
            if(node.next != null){minHeap.add(node.next);}
        }
        /* Process -
        for(iterate through the list a){
        if(list is not empty){its head node is addded to the minHeap}
        }
ListNode dummy = new ListNode(0); A dummy node is created to act as a placeholder for the beginning of the merged list
ListNode current = dummy; the current pointer is used to build the merged list
while(minHeap is not empty){
 ListNode node = minHeap.poll(); extract the smallest node from the heap
    current.next = node; add this node to the merged list
    current = current.next;
    if(extracted node has a nextNode){add the next node to the minHeap}
}
         */
        return dummy.next;// return the merged list starting from the node after the dummy node
    }

}


//* 42. Heaps - Build a Heap

class Solution {
    public int[] buildHeap(int[] A) {
        int n = A.length;

        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(A, n, i);
        }

        return A;
    }

    void heapify(int[] A, int n, int i) {
        int smallest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && A[left] < A[smallest]) {
            smallest = left;
        }

        if (right < n && A[right] < A[smallest]) {
            smallest = right;
        }

        if (smallest != i) {
            int temp = A[i];
            A[i] = A[smallest];
            A[smallest] = temp;

            heapify(A, n, smallest);
        }
    }
}




//* 43. Heaps - Ways to form Max Heap


public class Solution {
    private static final int MOD = 1000000007;

    // Precompute factorials and modular inverses
    private static long[] factorial;
    private static long[] inverseFactorial;

    static {
        int MAX = 100;
        factorial = new long[MAX + 1];
        inverseFactorial = new long[MAX + 1];

        factorial[0] = 1;
        for (int i = 1; i <= MAX; i++) {
            factorial[i] = factorial[i - 1] * i % MOD;
        }

        inverseFactorial[MAX] = modInverse(factorial[MAX], MOD);
        for (int i = MAX - 1; i >= 0; i--) {
            inverseFactorial[i] = inverseFactorial[i + 1] * (i + 1) % MOD;
        }
    }

    private static long modInverse(long a, long m) {
        return pow(a, m - 2, m);
    }

    private static long pow(long base, long exp, long mod) {
        long result = 1;
        while (exp > 0) {
            if (exp % 2 == 1) {
                result = result * base % mod;
            }
            base = base * base % mod;
            exp /= 2;
        }
        return result;
    }

    public int solve(int A) {
        // Dynamic Programming array to store the number of distinct max heaps for each size
        long[] dp = new long[A + 1];
        Arrays.fill(dp, 0);
        dp[0] = 1;
        dp[1] = 1;

        for (int i = 2; i <= A; i++) {
            dp[i] = calculateNumberOfHeaps(i, dp);
        }

        return (int) dp[A];
    }

    private long calculateNumberOfHeaps(int n, long[] dp) {
        if (n <= 1) return 1;

        // Compute number of nodes in the last level
        int h = (int) (Math.log(n + 1) / Math.log(2));
        int maxNodesInFullLevel = (1 << h) - 1;
        int totalNodesInLastLevel = n - maxNodesInFullLevel;
        int nodesInLastLevel = (1 << (h - 1));

        int leftNodes = maxNodesInFullLevel / 2;
        if (totalNodesInLastLevel > nodesInLastLevel) {
            leftNodes += totalNodesInLastLevel - nodesInLastLevel;
        }

        int rightNodes = n - 1 - leftNodes;

        long leftSubtreeWays = dp[leftNodes];
        long rightSubtreeWays = dp[rightNodes];
        long chooseLeftNodes = combination(n - 1, leftNodes);

        return leftSubtreeWays * rightSubtreeWays % MOD * chooseLeftNodes % MOD;
    }

    private long combination(int n, int k) {
        if (k > n) return 0;
        return factorial[n] * inverseFactorial[k] % MOD * inverseFactorial[n - k] % MOD;
    }
}




//* 44. Heaps - Product of 3


public class Solution {
    public int[] solve(int[] A) {
        int N = A.length;
        int[] B = new int[N];
        PriorityQueue<Integer> minHeap = new PriorityQueue<>();

        for (int i = 0; i < N; i++) {
            minHeap.add(A[i]);
            if (minHeap.size() > 3) {
                minHeap.poll();
            }
            if (i < 2) {
                B[i] = -1;
            } else {
                int[] topThree = new int[3];
                int index = 0;
                for (int num : minHeap) {
                    topThree[index++] = num;
                }
                B[i] = topThree[0] * topThree[1] * topThree[2];
            }
        }
        return B;
    }
}




//* 45. Heaps - Kth Smallest Element in a sorted matrix


public class Solution {
    public int solve(int[][] A, int B) {
        int N = A.length;
        int M = A[0].length;

        if (B > N * M) {
            throw new IllegalArgumentException("B is larger than the total number of elements in the matrix.");
        }

        PriorityQueue<int[]> minHeap = new PriorityQueue<>((a, b) -> Integer.compare(a[0], b[0]));

        for (int j = 0; j < M; j++) {
            minHeap.offer(new int[]{A[0][j], 0, j});
        }

        int[] directions = {0, 1};

        int result = -1;
        for (int i = 0; i < B; i++) {
            int[] current = minHeap.poll();
            result = current[0];
            int row = current[1];
            int col = current[2];

            if (row + 1 < N) {
                minHeap.offer(new int[]{A[row + 1][col], row + 1, col});
            }
        }

        return result;}
}



//46. Greedy - Flipkart's Challenge in Effective Inventory Management

public class Solution {
    public int solve(int[] A, int[] B) {
        int n = A.length;
        int mod = 1000000007;

        // Pairing expiration time and profit
        int[][] items = new int[n][2];
        for (int i = 0; i < n; i++) {
            items[i][0] = A[i];
            items[i][1] = B[i];
        }

        // Sorting items based on expiration time
        Arrays.sort(items, Comparator.comparingInt(item -> item[0]));

        // Max-heap to track the highest profits
        PriorityQueue<Integer> maxHeap = new PriorityQueue<>();

        int time = 0;
        long totalProfit = 0;

        for (int i = 0; i < n; i++) {
            if (time < items[i][0]) {
                maxHeap.add(items[i][1]);
                totalProfit += items[i][1];
                time++;
            } else if (!maxHeap.isEmpty() && maxHeap.peek() < items[i][1]) {
                totalProfit += items[i][1] - maxHeap.poll();
                maxHeap.add(items[i][1]);
            }
            totalProfit %= mod;
        }

        return (int) totalProfit;
    }
}



//* 47. Greedy - Finish Maximum Jobs

public class Solution {
    public int solve(int[] A, int[] B) {
        int n = A.length;

        // Pairing start and finish times
        int[][] jobs = new int[n][2];
        for (int i = 0; i < n; i++) {
            jobs[i][0] = A[i];
            jobs[i][1] = B[i];
        }

        // Sorting jobs based on finish time
        Arrays.sort(jobs, Comparator.comparingInt(job -> job[1]));

        int count = 1;  // Start with the first job
        int lastFinishTime = jobs[0][1];

        // Iterate through the sorted jobs
        for (int i = 1; i < n; i++) {
            if (jobs[i][0] >= lastFinishTime) {
                count++;
                lastFinishTime = jobs[i][1];
            }
        }

        return count;
    }
}





//* 48. Greedy - Distribute Candy

public class Solution {
    public int candy(int[] A) {
        int n = A.length;
        int[] candies = new int[n];

        // Initialize all children with 1 candy
        for (int i = 0; i < n; i++) {
            candies[i] = 1;
        }

        // Left to right pass
        for (int i = 1; i < n; i++) {
            if (A[i] > A[i - 1]) {
                candies[i] = candies[i - 1] + 1;
            }
        }

        // Right to left pass
        for (int i = n - 2; i >= 0; i--) {
            if (A[i] > A[i + 1]) {
                candies[i] = Math.max(candies[i], candies[i + 1] + 1);
            }
        }

        // Calculate the total number of candies
        int totalCandies = 0;
        for (int candy : candies) {
            totalCandies += candy;
        }

        return totalCandies;
    }
}




//* 49. Greedy - Another COin Problem

public class Solution {
    public int solve(int A) {
        int coins = 0;
        int coinValue = 1;

        while (A > 0) {
            coinValue = 1;
            while (coinValue * 5 <= A) {
                coinValue *= 5;
            }
            coins += A / coinValue;
            A %= coinValue;
        }

        return coins;
    }
}



//* 50. Greedy - Seats

public class Solution {
    public int seats(String A) {
        int MOD = 10000003;
        ArrayList<Integer> occupiedSeats = new ArrayList<>();

        // Step 1: Collect the indices of occupied seats ('x')
        for (int i = 0; i < A.length(); i++) {
            if (A.charAt(i) == 'x') {
                occupiedSeats.add(i);
            }
        }

        // If no one is sitting, no jumps are needed
        if (occupiedSeats.size() == 0) return 0;

        // Step 2: Find the median
        int n = occupiedSeats.size();
        int medianIndex = n / 2;
        int medianPosition = occupiedSeats.get(medianIndex);

        // Step 3: Calculate the minimum jumps required
        int minJumps = 0;
        for (int i = 0; i < n; i++) {
            int currentPosition = occupiedSeats.get(i);
            int targetPosition = medianPosition - medianIndex + i;
            minJumps = (minJumps + Math.abs(currentPosition - targetPosition)) % MOD;
        }

        return minJumps;
    }
}





//* 51. Greedy - Assign Mice to Holes


public class Solution {
    public int mice(int[] A, int[] B) {
        // Step 1: Sort both arrays
        Arrays.sort(A);
        Arrays.sort(B);

        int maxTime = 0;

        // Step 2: Calculate the maximum time
        for (int i = 0; i < A.length; i++) {
            maxTime = Math.max(maxTime, Math.abs(A[i] - B[i]));
        }

        return maxTime;

    }
}
