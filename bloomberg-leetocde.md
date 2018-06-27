## 116. Populating Next Right Pointers in Each Node
Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

Initially, all next pointers are set to NULL.

given

```
    1
   /  \
  2    3
 / \  / \
4  5  6  7

```

return

```
     1 -> NULL
   /  \
  2 -> 3 -> NULL
 / \  / \
4->5->6->7 -> NULL
```
```
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

    let queue = [root];

    while(queue.length) {

        let level = [];
        let size = queue.length;

        for (let i = 0; i < size; i++){
            let currentNode = queue.shift();
            level.push(currentNode)
            if (currentNode) queue.push(currentNode.left);
            if (currentNode) queue.push(currentNode.right);
        }

        for (let i = 0; i < level.length - 1; i++) {
            if (level[i]) level[i].next = level[i + 1] ? level[i + 1] : null;
        }
    }

};
```
