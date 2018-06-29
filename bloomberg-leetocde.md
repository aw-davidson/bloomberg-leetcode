## 117 Populating Next Right Pointers in Each Node II

Follow up for problem "Populating Next Right Pointers in Each Node".

What if the given tree could be any binary tree? Would your previous solution still work?

Note:

You may only use constant extra space.
For example,
Given the following binary tree,

```
         1
       /  \
      2    3
     / \    \
    4   5    7
```
After calling your function, the tree should look like:
```
         1 -> NULL
       /  \
      2 -> 3 -> NULL
     / \    \
    4-> 5 -> 7 -> NULL
```
```javascript
/**
 * Definition for binary tree with next pointer.
 * function TreeLinkNode(val) {
 *     this.val = val;
 *     this.left = this.right = this.next = null;
 * }
 */

/**
 * @param {TreeLinkNode} root
 * @return {void} Do not return anything, modify tree in-place instead.
 */

var connect = function(root) {

    if (!root) return;

    let queue = [root];

    while(queue.length) {

        let level = [];
        let size = queue.length;

        for (let i = 0; i < size; i++){
            let currentNode = queue.shift();
            level.push(currentNode)
            if (currentNode.left) queue.push(currentNode.left);
            if (currentNode.right) queue.push(currentNode.right);
        }

        for (let i = 0; i < level.length - 1; i++) {
            if (level[i]) level[i].next = level[i + 1] ? level[i + 1] : null;
        }
    }

};
```
Use a breadth-first traversal to gather all the nodes on one level in an array called level. The size of the queue at each iteration of the while loop corresponds to the number of nodes on the current level of the traversal.

## 160 Intersection of Two Linked Lists
Write a program to find the node at which the intersection of two singly linked lists begins.

For example, the following two linked lists:

```
A:          a1 → a2
                   ↘
                     c1 → c2 → c3
                   ↗
B:     b1 → b2 → b3
```
begin to intersect at node c1.

Notes:

If the two linked lists have no intersection at all, return `null`.
The linked lists must retain their original structure after the function returns.
You may assume there are no cycles anywhere in the entire linked structure.
Your code should preferably run in `O(n)` time and use only `O(1)` memory.

```javascript
/**
 * Definition for singly-linked list.
 * function ListNode(val) {
 *     this.val = val;
 *     this.next = null;
 * }
 */

/**
 * @param {ListNode} headA
 * @param {ListNode} headB
 * @return {ListNode}
 */
var getIntersectionNode = function(headA, headB) {

    let lengthA = 0;
    let lengthB = 0;
    let currentNode = headA;

    while (currentNode) {
        lengthA++;
        currentNode = currentNode.next;
    }

    currentNode = headB;

    while (currentNode) {
        lengthB++;
        currentNode = currentNode.next;
    }

    let larger = lengthA > lengthB ? headA : headB;
    let smaller = lengthA > lengthB ? headB : headA;

    let difference = Math.abs(lengthA - lengthB);

    let currentLarger = larger;

    while (difference) {
        currentLarger = currentLarger.next;
        difference--;
    }

    let currentSmaller = smaller;

    while(currentLarger && currentSmaller) {
        if (currentLarger === currentSmaller) return currentLarger;

        currentLarger = currentLarger.next;
        currentSmaller = currentSmaller.next;

    }

    return null;
};
```

Get the difference between the lengths of the linked lists. Iterate through the larger list by the difference. Compare nodes with the identity operator to find the intersection. O(n) times and 0(1) space.

## 122 Best Time to Buy and Sell Stock II
Say you have an array for which the ith element is the price of a given stock on day i.

Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).

```javascript
var maxProfit = function(prices) {
    let profit = 0;

    for (let i = 0; i < prices.length; i++) {
        if (prices[i+1]>prices[i]) profit += prices[i+1]-prices[i];
    }

    return profit;
};
```

If the next price is larger than the current price, add their difference to the profit.

## 53 Maximum Subarray

Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

For example, given the array `[-2,1,-3,4,-1,2,1,-5,4]`,
the contiguous subarray `[4,-1,2,1]` has the largest sum = `6`.

click to show more practice.

More practice:
If you have figured out the `O(n)` solution, try coding another solution using the divide and conquer approach, which is more subtle.


```javascript
/**
 * @param {number[]} nums
 * @return {number}
 */
var maxSubArray = function(nums) {

    let maxSum = nums[0];
    let localSum = 0;

    for (let i = 0; i < nums.length; i++) {


       //accumulate
        localSum += nums[i];

        //update maxSum if current num improves it
        maxSum = Math.max(localSum, maxSum);

        //reset local if it goes below 0
        localSum = Math.max(localSum, 0);
    }

    return maxSum;
};
```

## *69 Sqrt(x)

Implement `int sqrt(int x)`.

Compute and return the square root of `x`.

```javascript
/**
 * @param {number} x
 * @return {number}
 */
var mySqrt = function(x) {
    if (x < 1) return 0;

    let left = 1,
        right = x,
        mid;

    while (left <= right) {
        mid = Math.floor((right - left)/2 + left);

        if (mid > x / mid) { /* same as mid * mid > x but prevents overflow */
            right = mid - 1 /* answer <= (mid - 1) */
        } else {
            left = mid + 1
        }
    }

    return right;

};
```

Use a binary search to find the sqaure root. We find the last number for which (m * m <= x) is true.

## 42 Trapping Rain Water

Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

For example,
Given ```[0,1,0,2,1,0,1,3,2,1,2,1]```, return ```6```.

![enter image description here](http://www.leetcode.com/static/images/problemset/rainwatertrap.png)
The above elevation map is represented by array [0,1,0,2,1,0,1,3,2,1,2,1]. In this case, 6 units of rain water (blue section) are being trapped.


```javascript
/**
 * @param {number[]} height
 * @return {number}
 */
var trap = function(height) {

    let left = 0,
        right = height.length - 1,
        leftMax = 0,
        rightMax = 0,
        rainWater = 0;

    while (left < right) {
        if (height[left] < height[right]) {
            height[left] >= leftMax ? (leftMax = height[left]) : rainWater += (leftMax - height[left]);
            ++left;
        } else {
            height[right] >= rightMax ? (rightMax = height[right]) : rainWater += (rightMax - height[right]);
            --right;
        }
    }

    return rainWater;

};
```

Second Method:
```javascript
var trap = function(height) {

   let globalMax = height[0];
    let globalMaxIndex = 0;
    let rainWater = 0;

    for (let i = 1; i < height.length; i++){
        if (height[i] > globalMax) {
            globalMax = height[i];
            globalMaxIndex = i;
        }
    }

    let leftMax = 0;

    for (let i = 0; i < globalMaxIndex; i++){
        if (height[i] >= leftMax){
            leftMax = height[i];
        } else {
            rainWater += leftMax - height[i];
        }
    }

    let rightMax = 0;

    for (let i = height.length - 1; i > globalMaxIndex; i--){
        if (height[i] >= rightMax){
            rightMax = height[i];
        } else {
            rainWater += rightMax - height[i];
        }
    }

    return rainWater;

};
```
For the second method, we find the global max of the height array. Then we iterate through the array from left to the global max and from right to the global max. In each case the amount of rain water that is trapped is dictated by the preceding local maxes. On the left side of the global max the left wall will dictate the height of the rain water while on the right side the right wall will dictate. To calculate the water subtract the current height from the height of the previous max (the bounding wall other than the global max wall).

## 232 Implement Queue using Stacks
Implement the following operations of a queue using stacks.

`push(x)` -- Push element x to the back of queue.
`pop()` -- Removes the element from in front of queue.
`peek()`-- Get the front element.
`empty()` -- Return whether the queue is empty.
Notes:

 - You must use only standard operations of a stack -- which means only `push to top`, `peek/pop from top`, `size`, and `is empty` operations are valid.
 - Depending on your language, stack may not be supported natively. You may simulate a stack by using a list or deque (double-ended queue), as long as you use only standard operations of a stack.
 - You may assume that all operations are valid (for example,
   no pop or peek operations will be called on an empty queue).

```javascript
/**
 * Initialize your data structure here.
 */
var MyQueue = function() {
    this.stackA = [];
    this.stackB = [];
};

/**
 * Push element x to the back of queue.
 * @param {number} x
 * @return {void}
 */
MyQueue.prototype.push = function(x) {
    //always push onto stackA
    this.stackA.push(x)
};

/**
 * Removes the element from in front of queue and returns that element.
 * @return {number}
 */
MyQueue.prototype.pop = function() {
    if (this.stackB.length){
        return this.stackB.pop();
    } else {
        while (this.stackA.length) {
            this.stackB.push(this.stackA.pop());
        }
        return this.stackB.pop();
    }
};

/**
 * Get the front element.
 * @return {number}
 */
MyQueue.prototype.peek = function() {
    if (this.stackB.length){
        return this.stackB[this.stackB.length - 1];
    } else {
        while (this.stackA.length) {
            this.stackB.push(this.stackA.pop());
        }
        return this.stackB[this.stackB.length - 1];
    }
};

/**
 * Returns whether the queue is empty.
 * @return {boolean}
 */
MyQueue.prototype.empty = function() {
    return !this.stackA.length && !this.stackB.length;
};
```
