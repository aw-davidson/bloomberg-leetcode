## 116. Populating Next Right Pointers in Each Node

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
