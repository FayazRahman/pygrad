import autograd as ag

def is_number(s):
    try:
        float(s)
    except ValueError:
        return False
    return True

def parse(placeholders, fns, sub_exps):
    precedence = {
        '\n' : 0,
        ag.Add : 0,
        '-' : 0,
        ag.Multiply : 1,
        '/' : 1,
        ag.Pow : 2,
        ag.Graph : 3
    }

    last_func = None
    last_idx = 0
   
    inputs = [placeholders.copy()]
    
    dummies = inputs[0].copy()
    for sub_exp in sub_exps:
        input_idxs = slice(sub_exp[0], sub_exp[1] + 1)
        fn_idxs = slice(sub_exp[0], sub_exp[1])
        sub_graph, out = parse(dummies[input_idxs], fns[fn_idxs], [])
        fns[fn_idxs] = sub_graph,
    
    # inputs[0] = [inpt for inpt in inputs[0] if inpt != '\n']
    # print("inputs", inputs)
    
    while len(fns) > 0:
        done = []
        adj = 0

        if len(fns) == 1:
            if isinstance(fns[0], ag.Graph):
                inputs.append(fns[0](inputs[-1]))
            else:
                inputs.append(fns[0]()(inputs[-1]))
            fns.pop(0)
            ret = ag.Graph(placeholders, inputs[-1])
            ret.condense(inputs[-1])
            return ret, inputs[-1]

        for i in range(len(fns)):
            fn = fns[i]

            if last_func:
                
                if isinstance(last_func, ag.Graph):
                    layer = inputs[-1].copy()
                    layer[last_idx - adj : last_idx - adj + len(last_func.input_nodes)] = last_func(layer[last_idx - adj : last_idx - adj + len(last_func.input_nodes)]),
                    inputs.append(layer)
                    adj += 1
                    done.append(last_idx)

                elif precedence.get(fn) <= precedence.get(last_func):
                    layer = inputs[-1].copy()
                    layer[last_idx - adj : last_idx - adj + 2] = last_func()([layer[last_idx - adj], layer[last_idx - adj + 1]]),
                    inputs.append(layer)
                    adj += 1
                    done.append(last_idx)

                if fn == fns[-1]:
                    layer = inputs[-1].copy()
                    layer[i - adj : i - adj + 2] = fn()([layer[i - adj], layer[i - adj + 1]]),
                    inputs.append(layer)
                    adj += 1
                    done.append(i)
 
            last_func = fn
            last_idx = i
        
        last_func = None
        last_idx = 0
        
        for j in done[::-1]:
            fns.pop(j)
    
    ret = ag.Graph(placeholders, inputs[-1][0])
    ret.condense(inputs[-1][0])
    return ret, inputs[-1][0]

def tokenize(expression):
    fn = {
        '+' : ag.Add,
        '-' : None,
        '*' : ag.Multiply,
        '/' : None,
        '^' : ag.Pow
    }

    placeholders = []
    fns = []

    sub_level = 0
    sub_idx = []
    sub_exps = []

    tokens = expression.split()

    num_ph = -1

    for x in range(len(tokens)):
        
        token = tokens[x]
        
        if token in "+-/*^\n":
            fns.append(fn[token])

        elif not all(char in "+-/*^()\n" for char in token):
            num_ph += 1

            if is_number(token):
                placeholders.append(ag.PlaceHolder(float(token)))
            else:
                placeholders.append(ag.PlaceHolder())
        
        for char in token:
            if char == '(':
                sub_level += 1
                
                if char == token[0]:
                    if num_ph == -1:
                        sub_idx.append(0)
                    else:
                        sub_idx.append(num_ph)
                elif char == token[-1]:
                    sub_idx.append(num_ph + 1)
            
            if char == ')':
                assert sub_level > 0, "Check your expression!"
                sub_level -= 1

                if char == token[-1] and sub_idx[-1] != num_ph:
                    sub_exps.append((sub_idx[-1], num_ph))
                elif char == token[0] and sub_idx[-1] != num_ph - 1:
                    sub_idx.append((sub_idx[-1], num_ph - 1))

                sub_idx.pop()
    
    assert sub_level == 0, "Check your expression!"
    sub_exps = list(set(sub_exps))

    return placeholders, fns, sub_exps

placeholders, fns, sub_exps = tokenize(input("Enter expression: "))

graph, out = parse(placeholders, fns, sub_exps)

print(graph.f([2, 1, 2, 3, 5]))
