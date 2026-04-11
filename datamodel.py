import collections

Transition = collections.namedtuple('Transition',('state','action','next_state','reward'))

ActionRes = collections.namedtuple('ActionRes',('step','logits'))