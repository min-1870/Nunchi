"""hand labels for the 245-error sample of TR5n-102K (error_sample.jsonl, 2026-11-05), read with the model's output in
view: A = ambiguous with what the model sees (a prior / lean decides), B = set problem (garbage, wrong-mode text kept
as typed, obfuscated swearing, label that is not what the writer meant), C = context decides it for a person,
D = structure: the answer needs a split inside one word (fused, digits, typos made of jamo)"""
L = ("C C B B B B B D A B B D B B A A A B D A C C B C B B B B C C A B C C A C B "
     "A D A A A A A A A A A A A A A A A A A A D "  # 37-57
     "D A D D "  # 58-61
     "A A A A A A A A A A A A A A A "  # 62-76
     "D D D D D A D A A A "  # 77-86
     "A D A A A C A C C A A A B A B "  # 87-101
     "A D C C A D B B C D C B D D B D C C D B "  # 102-121
     "C C C C C C A C C C C C C C A "  # 122-136
     "B A A C C B C A B A A C B B A C A C A A "  # 137-156
     "B B A C B "  # 157-161
     "C C A A C A A A A C A C C C A C A A A C "  # 162-181
     "C C A C C C C C C A C C A A C "  # 182-196
     "D A A "  # 197-199
     "B D B D D D D D B D D D D D D B B B D D "  # 200-219 (20 items: 200..219)
     "C D C A D A D C C C D C A C C D A C D A "  # 220-239
     "D C D C D").split()  # 240-244
