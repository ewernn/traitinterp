/**
 * Conversation tree data structure for multi-turn chat with branching support.
 *
 * Each message is a node with parent-child relationships. Editing a previous
 * user message creates a sibling branch. The active path tracks which branch
 * is currently displayed.
 */

class ConversationNode {
    constructor({
        id,
        role,           // 'user' | 'assistant'
        content,        // Message text
        parentId,       // ID of parent node (null for root)
        tokenStartIdx,  // Global token index where this message starts
        tokenEndIdx,    // Global token index where this message ends
        tokenEvents,    // Array of {token, trait_scores} for assistant messages
        timestamp
    }) {
        this.id = id;
        this.role = role;
        this.content = content || '';
        this.parentId = parentId;
        this.tokenStartIdx = tokenStartIdx || 0;
        this.tokenEndIdx = tokenEndIdx || 0;
        this.tokenEvents = tokenEvents || [];
        this.timestamp = timestamp || Date.now();
        this.children = [];  // Array of child node IDs
    }
}

class ConversationTree {
    constructor() {
        this.nodes = new Map();      // id -> ConversationNode
        this.rootId = null;          // First message ID
        this.activePathIds = [];     // IDs from root to current leaf (active branch)
        this.globalTokens = [];      // Flat list of all tokens in active path
        this.messageRegions = [];    // [{messageId, startIdx, endIdx, role, content}, ...]
    }

    /**
     * Create a new message node and add it to the tree.
     */
    addMessage(role, content, parentId = null) {
        const id = this._generateId();
        const parent = parentId ? this.nodes.get(parentId) : null;

        // Calculate token start index based on parent's end
        const tokenStartIdx = parent ? parent.tokenEndIdx : 0;

        const node = new ConversationNode({
            id,
            role,
            content,
            parentId,
            tokenStartIdx,
            tokenEndIdx: tokenStartIdx  // Updated when tokens arrive
        });

        this.nodes.set(id, node);

        if (parent) {
            parent.children.push(id);
        } else {
            this.rootId = id;
        }

        // Update active path to include new node
        this._updateActivePath(id);

        return node;
    }

    /**
     * Append a token event to an assistant message during streaming.
     */
    appendToken(messageId, tokenEvent) {
        const node = this.nodes.get(messageId);
        if (!node || node.role !== 'assistant') return;

        node.tokenEvents.push(tokenEvent);
        node.tokenEndIdx = node.tokenStartIdx + node.tokenEvents.length;

        // Rebuild global state
        this._rebuildGlobalState();
    }

    /**
     * Finalize message content after streaming completes.
     */
    finalizeMessage(messageId, fullContent) {
        const node = this.nodes.get(messageId);
        if (node) {
            node.content = fullContent;
        }
    }

    /**
     * Get siblings of a node (for branch navigation).
     */
    getSiblings(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node || !node.parentId) return [node];

        const parent = this.nodes.get(node.parentId);
        return parent.children.map(id => this.nodes.get(id));
    }

    /**
     * Get index of node among its siblings.
     */
    getSiblingIndex(nodeId) {
        const node = this.nodes.get(nodeId);
        if (!node || !node.parentId) return 0;

        const parent = this.nodes.get(node.parentId);
        return parent.children.indexOf(nodeId);
    }

    /**
     * Switch to a different branch by navigating to a specific node.
     */
    switchBranch(nodeId) {
        this._updateActivePath(nodeId);
        this._rebuildGlobalState();
    }

    /**
     * Create a new branch by editing a user message.
     * Returns the new user node.
     */
    createBranch(originalMessageId, newContent) {
        const original = this.nodes.get(originalMessageId);
        if (!original || original.role !== 'user') return null;

        // Create new user message as sibling (same parent)
        const newUserNode = this.addMessage('user', newContent, original.parentId);

        // Recalculate token indices for the new branch
        // The new user message starts at the same place as the original
        newUserNode.tokenStartIdx = original.tokenStartIdx;
        newUserNode.tokenEndIdx = newUserNode.tokenStartIdx;  // User messages have no tokens

        return newUserNode;
    }

    /**
     * Get conversation history for API call (messages in active path).
     * If upToMessageId is provided, only includes messages up to (not including) that message.
     */
    getHistoryForAPI(upToMessageId = null) {
        const history = [];

        for (const id of this.activePathIds) {
            if (upToMessageId && id === upToMessageId) break;

            const node = this.nodes.get(id);
            if (node.content) {  // Skip empty messages
                history.push({
                    role: node.role,
                    content: node.content
                });
            }
        }

        return history;
    }

    /**
     * Get the last message ID in the active path.
     */
    getLastMessageId() {
        return this.activePathIds[this.activePathIds.length - 1] || null;
    }

    /**
     * Get a node by ID.
     */
    getNode(nodeId) {
        return this.nodes.get(nodeId);
    }

    /**
     * Check if tree has any messages.
     */
    isEmpty() {
        return this.nodes.size === 0;
    }

    /**
     * Clear the entire tree.
     */
    clear() {
        this.nodes.clear();
        this.rootId = null;
        this.activePathIds = [];
        this.globalTokens = [];
        this.messageRegions = [];
    }

    /**
     * Update active path to include a node and its ancestors/descendants.
     */
    _updateActivePath(nodeId) {
        // Build path from root to this node
        const pathToNode = [];
        let current = this.nodes.get(nodeId);

        while (current) {
            pathToNode.unshift(current.id);
            current = current.parentId ? this.nodes.get(current.parentId) : null;
        }

        // Extend to deepest child following existing active path or first child
        let leaf = this.nodes.get(nodeId);
        while (leaf && leaf.children.length > 0) {
            // If we have an existing path preference, follow it
            const activeChild = leaf.children.find(childId =>
                this.activePathIds.includes(childId)
            );
            const nextId = activeChild || leaf.children[0];
            leaf = this.nodes.get(nextId);
            pathToNode.push(leaf.id);
        }

        this.activePathIds = pathToNode;
    }

    /**
     * Rebuild globalTokens and messageRegions from active path.
     */
    _rebuildGlobalState() {
        this.globalTokens = [];
        this.messageRegions = [];

        let tokenIdx = 0;

        for (const id of this.activePathIds) {
            const node = this.nodes.get(id);
            const startIdx = tokenIdx;

            if (node.role === 'assistant' && node.tokenEvents.length > 0) {
                for (const event of node.tokenEvents) {
                    this.globalTokens.push(event);
                }
                tokenIdx += node.tokenEvents.length;
            }

            // Update node's token indices
            node.tokenStartIdx = startIdx;
            node.tokenEndIdx = tokenIdx;

            this.messageRegions.push({
                messageId: node.id,
                role: node.role,
                startIdx: startIdx,
                endIdx: tokenIdx,
                content: node.content
            });
        }
    }

    /**
     * Generate a unique ID.
     */
    _generateId() {
        return `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Serialize tree to JSON (for localStorage persistence).
     */
    toJSON() {
        return {
            nodes: Array.from(this.nodes.entries()).map(([id, node]) => ({
                id,
                role: node.role,
                content: node.content,
                parentId: node.parentId,
                tokenStartIdx: node.tokenStartIdx,
                tokenEndIdx: node.tokenEndIdx,
                tokenEvents: node.tokenEvents,
                timestamp: node.timestamp,
                children: node.children
            })),
            rootId: this.rootId,
            activePathIds: this.activePathIds
        };
    }

    /**
     * Restore tree from JSON.
     */
    fromJSON(data) {
        this.clear();

        for (const nodeData of data.nodes) {
            const node = new ConversationNode(nodeData);
            node.children = nodeData.children || [];
            this.nodes.set(nodeData.id, node);
        }

        this.rootId = data.rootId;
        this.activePathIds = data.activePathIds || [];
        this._rebuildGlobalState();
    }
}

// Export for use in other modules
window.ConversationNode = ConversationNode;
window.ConversationTree = ConversationTree;
