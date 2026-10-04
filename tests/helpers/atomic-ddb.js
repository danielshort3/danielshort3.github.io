'use strict';

// Minimal atomic store for exercising the admission behavior of the real SDK
// commands. Failed conditions leave every item untouched.
module.exports = function createAtomicStore() {
  const items = new Map();
  const key = (item) => `${item.pk}|${item.sk}`;
  return {
    items,
    async send(command) {
      await Promise.resolve();
      const input = command.input;
      if (input.Key) return { Item: items.get(key(input.Key)) };
      if (!Array.isArray(input.TransactItems)) throw new Error('A complete atomic admission transaction is required.');
      const reasons = input.TransactItems.map((operation) => {
        const update = operation.Update;
        const put = operation.Put;
        const item = items.get(key(put ? put.Item : update.Key));
        const values = update?.ExpressionAttributeValues || {};
        const failed = put ? Boolean(item)
          : (typeof values[':latest'] !== 'undefined' && Number(item?.lastQueryAt || 0) > values[':latest']) ||
            (typeof values[':limit'] !== 'undefined' && Number(item?.count || 0) >= values[':limit']);
        return { Code: failed ? 'ConditionalCheckFailed' : 'None' };
      });
      if (reasons.some((reason) => reason.Code !== 'None')) {
        const error = new Error('Admission rejected');
        error.name = 'TransactionCanceledException';
        error.CancellationReasons = reasons;
        throw error;
      }
      for (const operation of input.TransactItems) {
        if (operation.Put) {
          items.set(key(operation.Put.Item), { ...operation.Put.Item });
          continue;
        }
        const update = operation.Update;
        const values = update.ExpressionAttributeValues;
        const item = { ...update.Key, ...items.get(key(update.Key)), ttl: values[':ttl'] };
        if (typeof values[':one'] !== 'undefined') item.count = Number(item.count || 0) + values[':one'];
        if (update.UpdateExpression.includes('#lastQueryAt')) item.lastQueryAt = values[':now'];
        items.set(key(item), item);
      }
      return {};
    }
  };
};
