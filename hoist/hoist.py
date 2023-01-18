#!/usr/bin/env python3

# Copyright 2022- Norwegian Meteorological Institute

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sys import exit, argv
from pathlib import Path
from time import time

from loki import Sourcefile, Dimension, Subroutine, FindNodes, CallStatement
from loki import Loop, Transformer, as_tuple, SubstituteExpressions
from loki import FindVariables, Assignment, DeferredTypeSymbol, DerivedType
from loki import Quotient, Product, Sum, InlineCall
from loki import Scalar, Array, Import, VariableDeclaration, BasicType, LoopRange, RangeIndex
from loki import Pragma, Node, InternalNode
from loki import FloatLiteral, IntLiteral, LogicLiteral, LiteralList
from loki import CommentBlock, Comment, Module, Associate, Conditional, Section


def insert_routine_body(driver):
    '''
    driver: Subroutine object

    Replace calls to included routines with routine bodies
    '''

    minus_one = Product((-1, IntLiteral(1)))

    #List call objects and variable names in driver
    calls = FindNodes(CallStatement).visit(driver.body)
    d_names = [v.name for v in driver.variables]

    #Create ampty call map and kernel variable name list
    call_map = {}
    k_names = []

    #Loop over member subroutines
    for kernel in driver.members:

        #Make a set of kernel variables that are not arguments and 
        #not duplicates of driver variables
        kset = set(get_nonarguments(kernel)) - set(driver.variables)

        #Add names to l;ist of kernel variable names
        k_names += [v.name for v in kset]

        #Have to check for kernel variables with the same name as driver variables
        #Create sets of variables to add and remove create a map from old to new names
        pset = set()
        mset = set()
        name_map = {}

        #Loop over variables in kernel set and check if name exists in driver
        for v in kset:
            if v.name in d_names:

                #Add an X at the end of name until the name is unique
                new_name = v.name + 'X'
                while (new_name in d_names or new_name in k_names):
                    new_name += 'X'

                #Create variable with new name and organize set and map
                mset.add(v)
                pset.add(v.clone(name = new_name))
                k_names += [new_name]
                name_map[v.name] = new_name

        kset = kset - mset
        kset = kset.union(pset)

        #Check if any names must change
        kvar_map = {}
        if name_map:
            #Map variables to variables with new names
            for v in FindVariables(unique=False).visit(kernel.body):
                if v.name in name_map:
                    kvar_map[v] = v.clone(name = name_map[v.name])

        temp_body = SubstituteExpressions(kvar_map).visit(kernel.body)

        xvars = FindVariables().visit(temp_body)

        #Loop over all calls and check if they call the kernel
        for call in calls:
            if call.routine == kernel:

                #Create map from kernel dummy name to actual argument
                amap = {}
                for a in call.arg_iter():
                    amap[a[0].name] = a[1]

                #List kernel dummy variables
                variables = [v for v in FindVariables(unique=False).visit(temp_body) if v.name in amap]

                vmap = {}
                for v in variables:
                    #If actual argument not an array, just use it directly
                    if not isinstance(amap[v.name], Array):
                        vmap[v] = amap[v.name]
                    #If the shapes are the same, just use actual argument with kernel dimensions
                    elif isinstance(v, Array) and len(v.shape) == len(amap[v.name].shape):
                        vmap[v] = amap[v.name].clone(dimensions = v.dimensions)
                    else:
                        #Else we have to be careful
                        new_dims = []
                        ranges = sum(1 for d in amap[v.name].dimensions if isinstance(d, RangeIndex))

                        #If shape of dummy matches the number of ranges, match dimensions to ranges
                        if (len(v.shape) == ranges):

                            #Loop over dimensions of actual argument
                            j = 0
                            for a in amap[v.name].dimensions:
                                #If dimension is a range
                                if isinstance(a, RangeIndex):
                                    #If there's no lower range, just use kernel dimension, else subtract 1
                                    if not a.lower or (isinstance(a.lower, IntLiteral) and a.lower.value == 1):
                                        new_dims += [v.dimensions[j]]
                                    elif isinstance(a.lower, IntLiteral):
                                        new_dims += [Sum((IntLiteral(value = a.lower.value - 1), v.dimensions[j]))]
                                    else:
                                        new_dims += [Sum((a.lower, v.dimensions[j], minus_one))]
                                    j += 1

                                #else, just add actual argument dimension
                                else:
                                    new_dims += [a]

                        #If no ranges, first dimensions are from kernel, the rest are from driver
                        elif (ranges == 0):
                            new_dims += list(v.dimensions)
                            new_dims += list(amap[v.name].dimensions[len(v.shape):])

                        else:
                            raise Exception('Mismatch in dimensions')

                        vmap[v] = amap[v.name].clone(dimensions = as_tuple(new_dims))

                call_map[call] = SubstituteExpressions(vmap).visit(temp_body)

        driver.variables = as_tuple(list(driver.variables) + list(kset))
        d_names += k_names

    driver.body = Transformer(call_map).visit(driver.body)
    driver.contains = None


def is_comment(node):
    return isinstance(node, (Comment, CommentBlock))


def is_variable(node):
    return isinstance(node, (Scalar, Array, DeferredTypeSymbol))


def get_nonarguments(routine):
    return [v for v in routine.variables if v.name.lower() not in routine._dummies]


def get_routine_range(dimension, routine):
    '''
    routine: Subroutine object
    dimension: RangeIndex object

    Get routine variables and routine LoopRange
    '''

    rindex = routine.variable_map[dimension.index]

    if dimension.bounds[0].isnumeric():
        rstart = IntLiteral(dimension.bounds[0])
    else:
        rstart = routine.variable_map[dimension.bounds[0]]

    if dimension.bounds[1].isnumeric():
        rend = IntLiteral(dimension.bounds[1])
    else:
        rend = routine.variable_map[dimension.bounds[1]]

    rRange = LoopRange((rstart, rend))

    return rindex, rRange


def expression_variables(expression):
    """
    expression: Loki expression

    Recurse through expression, collect all the variables, and return as list.
    """

    variables = []

    #Add expression immediately if it is a scalar
    if isinstance(expression, Scalar):
        variables += [expression]

    #If expression is an array, add array variable itself and any dimension variables
    elif isinstance(expression, Array):
        variables += [expression.clone(dimensions=None)]
        for v in expression.dimensions:
            variables += expression_variables(v)

    #Add any variables if expression is a literal list
    elif isinstance(expression, LiteralList):
        for c in expression.elements:
            variables += expression_variables(c)

    #Add both numerator and denominator variables of quotients
    elif isinstance(expression, Quotient):
        variables += expression_variables(expression.numerator)
        variables += expression_variables(expression.denominator)

    #Add variables from any children in sums and products
    elif isinstance(expression, Product) or isinstance(expression, Sum):
        for c in expression.children:
            variables += expression_variables(c)

    #Add any parameters if expression is inlinecall
    elif isinstance(expression, InlineCall):
        for p in expression.parameters:
            variables += expression_variables(p)

    return variables


def remove_init_args(routine, variables):
    """
    Loop through variable declarations in routine and search for variables in inital expression.
    These variables are removed from the list of variables.
    """

    declarations = FindNodes(VariableDeclaration).visit(routine.spec)

    for declaration in declarations:
        for v in declaration.symbols:
            if v.type.initial:
                for ev in FindVariables().visit(v.type.initial):
                    if ev in variables:
                        variables.remove(ev)


def remove_body_vars(routine, variables):
    """
    Loop through variables in routine body.
    These are removed from the list of variables.
    """
    bvariables = FindVariables().visit(routine.body)

    names = [v.name for v in variables]
    for b in bvariables:
        if b.name in names:
            names.remove(b.name)

    for v in reversed(variables):
        if not v.name in names:
            variables.remove(v)


def remove_unused_arguments(driver, kernel):
    """
    driver: Subroutine object
    kernel: Subroutine object

    Removed unused arguments fro kernel and calls in driver
    """

    kargs = kernel.arguments
    kvars = kernel.variables

    #List arguments and remove those that appear in the argument inits and body
    uarguments = [a for a in kargs]
    remove_init_args(kernel, uarguments)
    remove_body_vars(kernel, uarguments)

    #Map the reverse arguments to their number in reverse
    amap = {}
    for a in reversed(uarguments):
        amap[a] = kargs.index(a)

    #List arguments and variables still not in uarguments
    new_arguments = [a for a in kargs if a not in uarguments]
    new_variables = [v for v in kvars if v not in uarguments]

    #List calls to kernel
    calls = [call for call in FindNodes(CallStatement).visit(driver.body) if call.name == kernel.name]

    #Loop over calls and map to new call without arguments
    #corresponding to unused kernel arguments
    cmap = {}
    for call in calls:

        cargs = call.arguments
        call_arguments = list(call.arguments)
        for a in zip(kargs, cargs):
            if a[0] in uarguments:
                call_arguments.remove(a[1])

        cmap[call] = call.clone(arguments = as_tuple(call_arguments))

    #Set kernel arguments and variables
    kernel.arguments = as_tuple(new_arguments) or None
    kernel.variables = as_tuple(new_variables) or None

    #Transform calls in driver body
    driver.body = Transformer(cmap).visit(driver.body)

def remove_unused_variables(routine):
    """
    routine: Subroutine object

    Remove unused variables from routine
    """

    #List routine variables not in arguments
    uvariables = get_nonarguments(routine)

    #Remove variables appearing in initializations
    remove_init_args(routine, uvariables)

    #Remove variables that appear in body
    remove_body_vars(routine, uvariables)

    #Set variables to variables that are not still in uvariables
    routine.variables = as_tuple([v for v in routine.variables if v not in uvariables]) or None


def constant_expression(expression, constants):
    """
    expression: Expression object
    constants: List of variables

    Determine if an expression is constant by recursion and until we reach a variable
    or a constant term.
    """

    #If expression is a FloatLiteral, an int or in the list of constants, it is constant
    if (isinstance(expression, FloatLiteral) or
        isinstance(expression, IntLiteral) or
        isinstance(expression, int) or expression in constants):
        return True

    #Check if all elements are constant in if expression is a list
    elif isinstance(expression, LiteralList):
        return all(constant_expression(c, constants) for c in expression.elements)

    #Check if both numerator and denominator are constant if expression is a quotient
    elif isinstance(expression, Quotient):
        return (constant_expression(expression.numerator, constants) and
                constant_expression(expression.denominator, constants))

    #If expression is a sum or product, check its children
    elif isinstance(expression, Product) or isinstance(expression, Sum):
        return all(constant_expression(c, constants) for c in expression.children)

    #Check parameters if it is an InlineCall
    elif isinstance(expression, InlineCall):
        return all(constant_expression(p, constants) for p in expression.parameters)

    #Else, return False
    return False


def parametrize(routine):
    """
    routine: Subroutine object

    Find variables in routine that are parameters and add the parameter keyword.
    """

    #List loop variables
    tracker = set()
    for l in FindNodes(Loop).visit(routine.body):
        tracker.add(l.variable)

    #Initialize dictionary and tracker and fill dictionary with existing parameters
    const_dict = {}
    for v in routine.variables:
        if v.type.parameter and not isinstance(v, Array):
            const_dict[v] = v.type.initial

    #Loop over all assignments
    all_assignments = iter(FindNodes(Assignment).visit(routine.body))
    for n in all_assignments:

        #If lhs of assignment is not in the tracker
        if not n.lhs in tracker:

            #Check if rhs is a constant expression
            if constant_expression(n.rhs, const_dict.keys()):

                #Add to dictionary and tracker if scalar or 1D fixed size array
                if (not isinstance(n.lhs, Array) or
                    (len(n.lhs.shape) == 1 and isinstance(n.lhs.shape[0], IntLiteral))):
                    const_dict[n.lhs] = n.rhs

        #If it is in tracker, remove it from the dictionary
        else:
            if n.lhs in const_dict:
                del const_dict[n.lhs]

        tracker.add(n.lhs)

    #Check arrays in the dictionary
    delete = set()
    modify = set()
    for c in const_dict:

        if isinstance(c, Array) and c not in delete and c not in modify:

            #List all instances in the dictionary with the same name as the array and their dimensions
            instances = []
            dims = []
            for t in const_dict:
                if t.name == c.name:
                    instances += [t]
                    dims += [t.dimensions]

            #If all instance lhs have dimensions
            if all(dims):

                #Check that the array appears the same number of times as the size, delete if not
                if (len(instances) != c.shape[0].value or
                    not all(isinstance(d[0], IntLiteral) for d in dims)):
                    delete.update(instances)

                #We'll have to construct the initial literal list
                else:
                    modify.update(instances)

            #else, check that array is only assigned once
            elif not any(dims):
                if len(instances) > 1:
                    delete.update(instances)

            #Delete if neither
            else:
                delete.update(instances)

    #Delete identified elements from dictionary
    for c in delete:
        del const_dict[c]

    #Generate a new dict for array initialization
    new_dict = {}
    for c in modify:

        length = c.shape[0].value
        index  = c.dimensions[0].value

        #Create new variable with dimensions equal to shape
        new_c = c.clone(dimensions = c.shape)
        if not new_c in new_dict:
            new_dict[new_c] = [None]*length

        #Add new dimension to new dict if index in range
        if 0 < index <= length:
            new_dict[new_c][index-1] = const_dict[c]

        #Remove c from original dict
        del const_dict[c]

    #If all dimensions are set, add to main dictionary
    for c in new_dict:
        if all(new_dict[c]):
            const_dict[c] = LiteralList(values = as_tuple(new_dict[c]))

    #List variable declarations and names of constants
    const_names = [c.name for c in const_dict]
    declarations = FindNodes(VariableDeclaration).visit(routine.spec)

    declaration_map = {}
    for d in declarations:

        variables = []

        #Check if any constants are declared
        any_constants = False
        for v in d.symbols:
            if v.name in const_names:
                any_constants = True
            else:
                variables += [v]

        #If there are constants, add to transformation map
        if any_constants:
            if variables:
                vdeclaration = d.clone(symbols = as_tuple(variables))
            else:
                vdeclaration = None
            declaration_map[d] = vdeclaration

    #Remove old declarations
    routine.spec = Transformer(declaration_map).visit(routine.spec)

    #Remove old assignments
    amap = {a : None for a in FindNodes(Assignment).visit(routine.body) if a.lhs.name in const_names}
    routine.body = Transformer(amap).visit(routine.body)

    #Add new declarations
    for c in const_dict:
        new_type = c.type.clone(parameter=True, initial=const_dict[c])
        new_c = c.clone(type = new_type)

        routine.spec.append(VariableDeclaration(symbols=(new_c,)))


def make_module(routine):
    """
    Create a module based on routine name.
    """

    module_name = routine.name + "_MOD"

    return Module(name = module_name)


def remove_associate(routine):
    """
    routine: Subroutine object

    Substitute all acssociated variables with their original variable,
    then replace the associate node with the associate body.
    """

    if routine.contains:
        for c in routine.contains.body:
            if isinstance(c, Subroutine):
                remove_associate(c)

    associate_map = {}
    associates = FindNodes(Associate).visit(routine.body)

    for a in associates:

        variables = FindVariables(unique=False).visit(a.body)
        inv = {v.name: k for k, v in a.association_map.items()}

        vmap = {}
        for v in variables:
            if v.name in inv:
                if isinstance(v, Array):
                    vmap[v] = inv[v.name].clone(dimensions=v.dimensions)
                else:
                    vmap[v] = inv[v.name].clone()

        associate_map[a] = SubstituteExpressions(vmap).visit(a.body)


    routine.body = Transformer(associate_map).visit(routine.body)


def pass_value(routine):
    """
    routine: Subroutine object

    Turn intent(in), scalar arguments into value arguments.
    """

    #Make scalar arguments with intent in into value arguments
    amap = {}
    for a in routine.arguments:
        if isinstance(a, Scalar) and isinstance(a.type.dtype, BasicType) and a.type.intent == 'in':
            new_type = a.type.clone(intent = None, value = True)
            amap[a] = a.clone(type = new_type)

    routine.spec = Transformer(amap).visit(routine.spec)

def split_loop(loop):
    '''
    loop: Loop object

    Split loop if possible
    '''

    loop_map = {}
    for b in loop.body:
        if isinstance(b, Loop):
            loop_map[b] = split_loop(b)

    new_loop = loop.clone(body = Transformer(loop_map).visit(loop.body))

    for b in new_loop.body:
        changed = []
        if isinstance(b, Assignment):
            changed += [b.lhs]
        elif isinstance(b, CallStatement): 
            for a in b.arg_iter():
                if (a[0].type.intent == 'out' or a[0].type.intent == 'inout'):
                    changed += [a[1]]
        

    return new_loop


def split_loops(routine):
    """
    routine: Subroutine object

    Find loops in routine body and split them if possible.
    """
    
    outer_loops = list_outer_loops(routine.body.body)

    loop_map = {}
    for loop in outer_loops:
        loop_map[loop] = split_loop(loop)

    new_routine = routine.clone(body = Transformer(loop_map).visit(routine.body))

    return new_routine


def find_next(loop, nodes):
    """
    Determine if the next loop has the same bounds and is not separated by anything by a comment.
    If so, return a list of all loops that qualifies.
    """

    candidates = []
    i = len(FindNodes(Node).visit(loop.body)) + 1
    while i < len(nodes):
        n = nodes[i]
        if isinstance(n,Loop) and (n.variable == loop.variable.name and
                                   n.bounds.start == n.bounds.start.name and
                                   n.bounds.stop == n.bounds.stop.name):
            candidates = [n]
            candidates += find_next(n, nodes[i:])
            break
        elif not is_comment(n,):
            break
        i += 1

    return candidates


def find_candidates(dimension, section):
    """
    Find candidate loops that have the same dimensions and might be merged.
    """

    nodes = FindNodes(Node).visit(section)
    i = 0
    candidates = []
    while i < len(nodes):
        n = nodes[i]
        if hasattr(n, 'body') and n.body:
            if isinstance(n, Loop) and n.variable == dimension.index:
                c = [n]
                c += find_next(n, nodes[i:])
                if len(c) > 1:
                    candidates += [as_tuple(c)]
                i = nodes.index(FindNodes(Node).visit(c[-1])[-1]) + 1
            else:
                candidates += find_candidates(dimension, n.body)
                i += len(FindNodes(Node).visit(n))
        else:
            i += 1

    return candidates


def merge_loops(dimension, routine):
    """
    Merge together consecutive loops that can be merged.
    """

    candidates = find_candidates(dimension, routine)

    call_map = {}
    for c in candidates:
        x = c[0].body
        for l in c[1:]:
            x += l.body
            call_map[l] = None

        call_map[c[0].body] = x

    routine.body = Transformer(call_map).visit(routine.body)


def list_outer_loops(body_tuple):
    '''
    body_tuple: tuple of nodes
    outer_loops: list to add loops to

    Recurse through body tuples and return a list of outer loops
    '''


    outer_loops = []
    for b in body_tuple:
        if isinstance(b, Loop):
            outer_loops += [b]
        elif isinstance(b, InternalNode):
            outer_loops += list_outer_loops(b.body)
            if isinstance(b, Conditional):
                outer_loops += list_outer_loops(b.else_body)

    return outer_loops


def move_independent_loop_out(indydim, routine):
    '''
    indydim: RangeIndex object
    routine: Subroutine object

    Find any loops with index indydim andmake them the outer loop if they are nested.
    '''

    loops = list_outer_loops(routine.body.body)

    loops = [l for l in loops if l.variable != indydim.index]

    rindex, rRange = get_routine_range(indydim, routine)

    loop_map = {}
    for l in loops:

        bodymap = {il: il.body for il in FindNodes(Loop).visit(l.body) if il.variable == indydim.index}

        if bodymap:

            new_body = l.clone(body = Transformer(bodymap).visit(l.body))
            loop_map[l] = Loop(rindex, bounds = rRange, body = new_body)

    routine.body = Transformer(loop_map).visit(routine.body)


def remove_hook(routine):
    '''
    routine: Subroutine object

    Remove calls to dr_hook
    '''

    none_map = {}
    for c in FindNodes(Conditional).visit(routine.body):
        if c.condition == 'LHOOK':
            none_map[c] = None
        
    routine.body = Transformer(none_map).visit(routine.body)


def add_gang(routine, dimension):
    '''
    routine: Subroutine object
    dimension: RangeIndex object

    Add acc loop gang to loops in routine matching dimensions.
    '''

    loops = [l for l in FindNodes(Loop).visit(routine.body) if l.variable == dimension.index]

    loop_map = {}
    for l in loops:
        loop_map[l] = l.clone(pragma = Pragma(keyword = 'acc', content='loop gang'))

    routine.body = Transformer(loop_map).visit(routine.body)


def add_vector(routine, dimension):
    '''
    routine: Subroutine object
    dimension: RangeIndex object

    Add acc loop vector to loops in routine matching dimensions.
    '''

    loops = [l for l in FindNodes(Loop).visit(routine.body) if l.variable == dimension.index]

    loop_map = {}
    for l in loops:
        loop_map[l] = l.clone(pragma = Pragma(keyword = 'acc', content='loop vector'))

    routine.body = Transformer(loop_map).visit(routine.body)


def add_sequential(routine, dimension):
    '''
    routine: Subroutine object
    dimension: RangeIndex object

    Add acc loop seq to loops in routine matching dimensions.
    '''

    loops = [l for l in FindNodes(Loop).visit(routine.body) if l.variable == dimension.index]

    loop_map = {}
    for l in loops:
        loop_map[l] = l.clone(pragma = Pragma(keyword = 'acc', content='loop seq'))

    routine.body = Transformer(loop_map).visit(routine.body)


def add_parallel_node(node, mapping):
    '''
    routine: Subroutine object
    '''

    if hasattr(node, 'pragma') and node.pragma:
        new_pragma = node.pragma.clone(content = 'parallel ' + node.pragma.content)
        mapping[node] = node.clone(pragma = new_pragma)
    else:
        if hasattr(node, 'body'):
            for n in node.body:
                add_parallel_node(n, mapping)

        if hasattr(node, 'else_body'):
            for n in node.else_body:
                if isinstance(n, tuple):
                    for x in n:
                        add_parallel_node(x, mapping)
                else:
                    add_parallel_node(n, mapping)


def add_parallel(routine):
    '''
    routine: Subroutine object
    '''

    mapping = {}
    for node in routine.body.body:
        add_parallel_node(node, mapping)

    routine.body = Transformer(mapping).visit(routine.body)

def add_acc(routine, gang=None, vector=None, sequential=None):
    '''
    routine: Subroutine object
    gang: RangeIndex object
    vector: RangeIndex object
    sequential: RangeIndex object

    '''

    if gang:
        add_gang(routine, gang)

    if vector:
        add_vector(routine, vector)

    if sequential:
        add_sequential(routine, sequential)

    add_parallel(routine)


def add_seq(routines):
    '''
    routines: list of Subroutine object

    Add openacc sequential to the beginning of routine.
    '''

    seq_pragma = Pragma(keyword='acc', content='routine seq')
    for routine in routines:
        routine.body.prepend(seq_pragma)


def add_data(routine):
    '''
    Add data statements to routines
    '''

    ivars = set()
    ovars = set()

    loops = [l for l in FindNodes(Loop).visit(routine.body) if l.pragma]

    loop_vars = set()
    for l in loops:
        loop_vars.add(l.variable)
    end = time()

    for l in loops:

        assignments = FindNodes(Assignment).visit(l.body)
        for a in assignments:

            if (isinstance(a.lhs, Array) or isinstance(a.lhs.type.dtype, DerivedType)) and a.lhs not in loop_vars and not a.lhs.type.parameter:
                if a.lhs.parent:
                    ovars.add(a.lhs.parent)
                else:
                    ovars.add(a.lhs.clone(dimensions=None))

            for v in FindVariables().visit(a.rhs):
                if (isinstance(v, Array) or isinstance(v.type.dtype, DerivedType)) and v not in loop_vars and not v.type.parameter:
                    if v.parent:
                        ivars.add(v.parent)
                    else:
                        ivars.add(v.clone(dimensions=None))

        calls = FindNodes(CallStatement).visit(l.body)
        for c in calls:

            if c.routine is not BasicType.DEFERRED:
                for a in c.arg_iter():

                    if is_variable(a[1]):
                        if (isinstance(a[1], Array) or isinstance(a[1].type.dtype, DerivedType)) and a[1] not in loop_vars and not a[1].type.parameter:

                            if (a[0].type.intent == 'inout' or a[0].type.intent == 'in' or a[0].type.value):
                                ivars.add(a[1].clone(dimensions=None))

                            if (a[0].type.intent == 'inout' or a[0].type.intent == 'out'):
                                ovars.add(a[1].clone(dimensions=None))

    iovars = ivars.intersection(ovars)
    ivars = ivars - iovars
    ovars = ovars - iovars

    cvars = set()

    for i in ivars:
        if not (i.type.intent or i.type.value):
            cvars.add(i)

    for o in ovars:
        if not (o.type.intent or o.type.value):
            cvars.add(o)

    for io in iovars:
        if not (io.type.intent or io.type.value):
            cvars.add(io)

    iovars = iovars - cvars
    ivars = ivars - cvars
    ovars = ovars - cvars

    content = "data "

    if ovars:
        out_content = 'copyout('
        for o in ovars:
            out_content += (o.name + ', ')
        out_content = out_content[:-2] + ') '
        content += out_content

    if ivars:
        in_content = 'copyin('
        for i in ivars:
            in_content += (i.name + ', ')
        in_content = in_content[:-2] + ') '
        content += in_content

    if iovars:
        inout_content = 'copy('
        for io in iovars:
            inout_content += (io.name + ', ')
        inout_content = inout_content[:-2] + ') '
        content += inout_content

    if cvars:
        c_content = 'create('
        for c in cvars:
            c_content += (c.name + ', ')
        c_content = c_content[:-2] + ') '
        content += c_content

    data_pragma = Pragma(keyword = "acc", content = content)
    end_pragma = Pragma(keyword = "acc", content = 'end data')

    routine.spec.append(data_pragma)
    routine.body.append(end_pragma)
    end = time()


def reorder_arrays(driver, kernels):
    '''
    driver: Subroutine object
    kernels: tuple of Subroutine objects

    Reorder array dimensions in kernels so they match dimensions in driver.
    '''


    #List all calls in driver
    calls = FindNodes(CallStatement).visit(driver.body)

    #Get a list of the names of driver arguments
    driver_args = [a.name for a in driver.arguments]

    #Map calls to kernels
    kcalls = {}
    for kernel in kernels:
        kcalls[kernel] = [call for call in calls if call.routine == kernel]

    shuffleargs = {}

    #Loop over kernels and the calls to the kernel
    for kernel in kcalls:

        for c in kcalls[kernel]:
            #a[0] is the variable in kernel
            #a[1] is the variable in driver
            for a in c.arg_iter():
                if isinstance(a[0], Array) and isinstance(a[1], Array) and a[1].name not in driver_args:

                    #If shape length mismatch
                    if len(a[0].shape) != len(a[1].shape):

                        r_index = []
                        s_index = []
                        for i, d in enumerate(a[1].dimensions):
                            if isinstance(d, RangeIndex):
                                r_index += [i]
                            else:
                                s_index += [i]

                        #Full dimensions index
                        new_index = r_index + s_index

                        #Check that the list of range indices has the correct length
                        if len(r_index) != len(a[0].shape):
                            raise Exception('Number of range indices not the same as shape length')

                        if a[1].name in shuffleargs:
                            if not new_index == shuffleargs[a[1].name]:
                                raise Exception('Different shapes in different kernels')

                        #Store new_index and variable
                        shuffleargs[a[1].name] = new_index

    #List variables showing up in shuffleargs
    dvars = [v for v in driver.variables if v.name in shuffleargs]

    #Generate map to new versions with reordered shape and dimensions
    vmap = {}
    for v in dvars:
        new_shape = reduced_list(v.shape, shuffleargs[v.name])
        new_dims  = reduced_list(v.dimensions, shuffleargs[v.name])
        new_type = v.type.clone(shape = new_shape)
        vmap[v] = v.clone(dimensions = new_dims, type = new_type)

    #Transform variable shapes
    driver.spec = SubstituteExpressions(vmap).visit(driver.spec)

    #List variables in driver body in shuffleargs that use dimensions
    bvars = [v for v in FindVariables(unique=False).visit(driver.body)
             if v.name in shuffleargs and v.dimensions]

    #Generate map to variables with reordered dimensions
    vmap = {}
    for v in bvars:
        new_dims = reduced_list(v.dimensions, shuffleargs[v.name])
        vmap[v] = v.clone(dimensions = new_dims)

    #Transform driver body
    driver.body = SubstituteExpressions(vmap).visit(driver.body)


def reorder_arrays_dim(routine, dimension):
    '''
    routine: Subroutine object
    dimension: RangeIndex object

    Reorder array shape so dimension becomes the last
    '''


    #Get the routine variable corresponding to dimension size
    size = routine.variable_map[dimension.size]

    #Create a list of local arrays that are not arguments and have size in their shape
    local_arrays = []
    for v in get_nonarguments(routine):
        if isinstance(v, Array) and size in v.shape:
            local_arrays += [v]
    end = time()

    #Create a dict matching array name to index of size in shape
    l_arrays = {}
    for a in local_arrays:

        indices = [i for i,s in enumerate(a.shape) if s == size]

        j = 1
        for i in reversed(indices):
            if i != len(a.shape) - j:
                l_arrays[a.name] = indices
                break
            else:
                j += 1

    #Remove any variables where size dimension is passed to subroutines
    for call in FindNodes(CallStatement).visit(routine.body):
        for a in call.arguments:
            if is_variable(a) and a.name in l_arrays:
                for i in l_arrays[a.name]:
                    if not a.dimensions or isinstance(a.dimensions[i], RangeIndex):
                        del l_arrays[a.name]

    #Generate map to transform routine spec
    spec_map = {}
    for v in FindVariables().visit(routine.spec):
        if v.name in l_arrays:
            ind = [i for i in range(len(v.shape)) if i not in l_arrays[v.name]]
            ind += l_arrays[v.name]

            new_type = v.type.clone(shape = reduced_list(v.shape, ind))
            new_dims = reduced_list(v.dimensions, ind)

            spec_map[v] = v.clone(dimensions = new_dims, type = new_type)

    #Transform routine spec
    routine.spec = SubstituteExpressions(spec_map).visit(routine.spec)

    #Generate map for routine body
    body_map = {}
    for v in FindVariables(unique=False).visit(routine.body):
        if v.name in l_arrays:
            ind = [i for i in range(len(v.shape)) if i not in l_arrays[v.name]]
            ind += l_arrays[v.name]
            new_dims = reduced_list(v.dimensions, ind)

            body_map[v] = v.clone(dimensions = new_dims)

    #Transform routine body
    routine.body = SubstituteExpressions(body_map).visit(routine.body)


def add_loops(driver, kernel, loop_range):
    """
    driver: Subroutine object
    kernel: Subroutine object
    loop_range: RangeIndex object

    Add loops based on loop_range around the calls to kernel in driver.
    """

    #Get the variables from loop_range
    dindex, dRange = get_routine_range(loop_range, driver)

    #Loop over call to kernel and generate map to call inside loop
    call_map = {}
    for call in FindNodes(CallStatement).visit(driver.body):
        if call.name == kernel.name:
            call_map[call] = Loop(variable=dindex, bounds=dRange, body=[call])

    driver.body = Transformer(call_map).visit(driver.body)


def remove_loops(routine, loop_range):
    """
    routine: Subroutine object
    loop_range: RangeIndex object

    Remove loops with loop_range from routine.
    """

    #If loop variable is the same as in range, map the loop to its body.
    loop_map = {}
    for loop in FindNodes(Loop).visit(routine.body):
        if loop.variable == loop_range.index:
            loop_map[loop] = loop.body

    routine.body = Transformer(loop_map).visit(routine.body)


def hoist_loops(driver, kernel, loop_range):
    """
    driver: Subroutine object
    kernel: Subroutine object
    loop_range: RangeIndex object

    Remove loops based on loop_range from kernel and
    add the loops around the calls to kernel in driver.
    """

    add_loops(driver, kernel, loop_range)

    remove_loops(kernel, loop_range)

    demote_arguments(driver, kernel, loop_range)

    demote_variables(kernel, loop_range)


def reduced_list(old_tuple, indices):
    """
    old_list: list
    indices: list of indices of the elements to keep

    return a new_list only containing the elements with index in indices
    """

    new_tuple = []
    for i in indices:
        new_tuple += [old_tuple[i]]

    return as_tuple(new_tuple) or None


def demote_arguments(driver, kernel, dimension):
    """
    driver: Subroutine object
    kernel: Subroutine object
    dimension: RangeIndex

    Demote variables in kernel that have dimension in shape
    and modify the corresponding arguments in call
    """

    kvars = FindVariables(unique=False).visit(kernel.body)

    amap = {}
    vmap = {}
    arg_index = []
    for j, a in enumerate(kernel.arguments):

        if isinstance(a, Array):

            #Generate new shape and dimension tuples by removing shape variables
            #corresponding to  the size of dimensions
            new_shape = []
            new_dims  = []

            #Loop over shape and dimension
            dims = []
            for i, s in enumerate(a.shape):
                if s not in dimension.size_expressions:
                    new_shape += [s]
                    dims += [i]

            new_shape = as_tuple(new_shape) or None

            #If new_shape is not the same as a.shape, add to mapping
            if new_shape != a.shape:

                new_type = a.type.clone(shape=new_shape)
                amap[a] = a.clone(dimensions=new_shape, type=new_type)
                arg_index += [j]

                avars = [v for v in kvars if v.name == a.name]

                #Generate new dimensions based on dims and create a map
                for v in avars:
                    new_dims = reduced_list(v.dimensions, dims)
                    vmap[v] = v.clone(dimensions=new_dims, type=new_type)

    #List the calls to kernel in driver
    call_list = [c for c in FindNodes(CallStatement).visit(driver.body) if c.name == kernel.name]

    #Find loops in driver with dimension index as index
    loops = [l for l in FindNodes(Loop).visit(driver.body) if l.variable==dimension.index]

    #Map calls in loops to the loop index
    varmap = {}
    for l in loops:
        lcalls = [c for c in FindNodes(CallStatement).visit(l.body) if c in call_list]
        for c in lcalls:
            varmap[c] = l.variable
            call_list.remove(c)

    #Calls outside loops use the first element
    for c in call_list:
        varmap[c] = IntLiteral('1')

    #Loop over the relevant calls and the arguments that involve dimension
    cmap = {}
    for c in varmap:
        for i in arg_index:

            a = c.arguments[i]
            new_dims = []

            #If argument has dimensions, use them, but remove relevant dimension
            if a.dimensions:
                for d,s in zip(a.dimensions, a.shape):
                    if s in dimension.size_expressions:
                        new_dims += [varmap[c]]
                    else:
                        new_dims += [d]

            #Otherwise pass the whole dimension
            else:
                for s in a.shape:
                    if s in dimension.size_expressions:
                        new_dims += [varmap[c]]
                    else:
                        if isinstance(s, RangeIndex):
                            new_dims += [s]
                        else:
                            new_dims += [RangeIndex((IntLiteral(1), s))]

            #Make a map for the arguments
            new_dims = as_tuple(new_dims) or None
            cmap[a] = a.clone(dimensions = new_dims)

    #Transform kernel spec and body and driver
    kernel.spec = SubstituteExpressions(amap).visit(kernel.spec)

    kernel.body = SubstituteExpressions(vmap).visit(kernel.body)

    driver.body = SubstituteExpressions(cmap).visit(driver.body)
    


def demote_variables(routine, dimension):
    """
    routine: Subroutine object
    dimension: RangeIndex object

    Demote any variables where dimension appears in shape.
    """

    #List array variables with dimension in shape that are not arguments
    variables = get_nonarguments(routine)
    variables = [v for v in variables if isinstance(v, Array)]
    variables = [v for v in variables if any(d in dimension.size_expressions for d in v.shape)]

    #Construct maps to the variables to their new versions
    #and keep a dict of indices removed.
    varmap = {}
    vmap = {}
    for var in variables:
        dims = []
        new_shape = []
        new_dims = []
        for i, (s, d) in enumerate(zip(var.shape, var.dimensions)):
            if s not in dimension.size_expressions:
                dims += [i]
                new_shape += [s]
                new_dims += [d]
        varmap[var.name] = dims

        new_type = var.type.clone(shape=new_shape or None)
        new_dims = as_tuple(new_dims) or None
        vmap[var] = var.clone(dimensions=new_dims, type=new_type)

    #Transform routine spec
    routine.spec = SubstituteExpressions(vmap).visit(routine.spec)

    #Find body variables present in varmap
    bvars = FindVariables(unique=False).visit(routine.body)
    bvars = [v for v in bvars if v.name in varmap]

    #Create map to new dimensions
    vmap = {}
    for var in bvars:
        new_dims = []
        for i in varmap[var.name]:
            new_dims += [var.dimensions[i]]
        vmap[var] = var.clone(dimensions=as_tuple(new_dims) or None)

    #Transform routine body
    routine.body = SubstituteExpressions(vmap).visit(routine.body)


def remove_index_arg(driver, kernel, index):
    '''
    driver: Subroutine object
    kernel: Subroutine object
    index: string

    Remove the index argument from kernel and remove it from calls in driver.
    '''

    #List array variables in kernel
    variables = [v for v in FindVariables(unique=False).visit(kernel.body) if isinstance(v, Array)]

    #Map variable names to position of index in dimensions
    imap = {}
    for v in variables:
        if v.dimensions and index in v.dimensions:
            imap[v.name] = v.dimensions.index(index)

    #Modify the variables kernel
    vmap = {}
    bmap = {}
    for v in kernel.variables:
        if v.name in imap:

            new_shape = v.shape[:imap[v.name]] + v.shape[imap[v.name]+1:]
            new_shape = as_tuple(new_shape) or None
            new_type = v.type.clone(shape=new_shape)

            new_dims = v.dimensions[:imap[v.name]] + v.dimensions[imap[v.name]+1:]
            new_dims = as_tuple(new_dims) or None

            vmap[v] = v.clone(dimensions = new_dims, type = new_type)

    #Modify the kernel spec
    kernel.spec = SubstituteExpressions(vmap).visit(kernel.spec)

    #Map variables in body to new dimensions
    for b in variables:
        if b.name in imap:
            new_dims = b.dimensions[:imap[b.name]] + b.dimensions[imap[b.name]+1:]
            bmap[b] = b.clone(dimensions = (as_tuple(new_dims) or None))

    #Transform body variables
    kernel.body = SubstituteExpressions(bmap).visit(kernel.body)

    #List calls to kernel
    calls = [c for c in FindNodes(CallStatement).visit(driver.body) if c.name == kernel.name]

    kargs = kernel.arguments

    cmap = {}
    for c in calls:

        #Get call arguments
        cargs = c.arguments

        #Find the index variable in driver
        driver_index = None
        for a in zip(kargs, cargs):
            if a[0] == index:
                driver_index = a[1]
                break

        #Loop over the arguments in call and modify the dimensions
        new_args = []
        for a in zip(kargs, cargs):

            #Check that it is an array in driver and check that name is in imap
            if isinstance(a[1], Array) and a[1].name in imap:

                i=0
                new_dims = []
                if a[1].dimensions:
                    #Loop over dimensions in driver and
                    #find the index in driver that corresponds to the index in kernel
                    #by counting RangeIndex dimensions.
                    for d in a[1].dimensions:
                        if isinstance(d, RangeIndex):
                            if i == imap[a[1].name]:
                                new_dims += [driver_index]
                            else:
                                new_dims += [d]
                            i += 1
                        else:
                            new_dims += [d]
                else:
                    #If no dimensions, insert shape as dimensions except index
                    for i,d in enumerate(a[1].shape):
                        if i == imap[a[1].name]:
                            new_dims += [driver_index]
                        else:
                            if isinstance(d, RangeIndex):
                                new_dims += [d]
                            else:
                                new_dims += [RangeIndex((IntLiteral(1), d))]

                new_dims = as_tuple(new_dims) or None
                new_arg  = a[1].clone(dimensions = new_dims)
                new_args += [new_arg]

            elif a[0] != index:
                new_args += [a[1]]

        cmap[c] = c.clone(arguments = as_tuple(new_args))

    #Transform calls in driver body
    driver.body = Transformer(cmap).visit(driver.body)

    #Remove index from kernel arguments and variables
    i = kargs.index(index)

    kernel.arguments = kargs[:i] + kargs[i+1:]

    kvars = kernel.variables

    i = kvars.index(index)

    kernel.variables = kvars[:i] + kvars[i+1:]


def pass_undefined(driver, kernel):
    '''
    driver: Subroutine object
    kernel: Subroutine object

    Add arguments to kernel that are not passed from driver because kernel is inside driver.
    '''

    #List driver variables that appear in kernel, but are not listed in kernel variables
    var_names = [v.name for v in kernel.variables]
    u_names = [v.name for v in FindVariables(unique=True).visit(kernel.body) if v.name not in var_names]
    undefined = (d for d in driver.variables if d.name in u_names)

    #Strip away any dimensions in array variables
    new_arguments = []
    shapevars = set()
    for var in undefined:
        if isinstance(var, Array):
            new_arguments += [var.clone(dimensions=None)]
            for s in var.shape:
                if isinstance(s, Scalar):
                    shapevars.add(s)
                elif isinstance(s, RangeIndex):
                    for v in FindVariables().visit(s):
                        shapevars.add(v)
        else:
            new_arguments += [var.clone()]

    #Add any collected shape variables
    for v in shapevars:
        if v.name not in kernel.variables:
            new_arguments += [v]

    #Create a map of calls in driver mapping to new calls with new arguments
    cmap = {}
    for c in FindNodes(CallStatement).visit(driver.body):
        if c.name == kernel.name:
            cmap[c] = c.clone(arguments = c.arguments + as_tuple(new_arguments))

    #Generate kernel versions of the new variables
    for i,a in enumerate(new_arguments):
        new_type = a.type.clone(intent='in', parameter=None, initial=None)
        if isinstance(a, Array):
            new_arguments[i] = a.clone(scope=kernel, type=new_type, dimensions=a.shape)
        else:
            new_arguments[i] = a.clone(scope=kernel, type=new_type)

    #Add the new variables to kernel arguments
    kernel.arguments = kernel.arguments + tuple(new_arguments)

    #Create a map mapping the variables to their kernel versions
    kmap = {}
    for v in FindVariables(unique=False).visit(kernel.body):
        for na in new_arguments:
            if v.name == na.name:
                if isinstance(na, Array):
                    kmap[v] = na.clone(dimensions=v.dimensions)
                else:
                    kmap[v] = na
                break

    #Transform the routine bodies
    driver.body = Transformer(cmap).visit(driver.body)
    kernel.body = SubstituteExpressions(kmap).visit(kernel.body)


def insert_imports(driver, kernel):
    '''
    Insert imports from driver that are missing in kernel.
    '''

    #First, list variables used in declaration types
    declarations = list(FindNodes(VariableDeclaration).visit(kernel.spec))
    undefined = []
    for D in declarations:

        #Get the type of the variables being declared
        t = D.symbols[0].type

        #If symbols are of basic type and have a name, add name to list of undefined
        if isinstance(t.dtype, BasicType):
            if t.kind and t.kind.name not in undefined:
                undefined += [t.kind.name]

        #If the type has a name, add it to undefined immediately
        elif t.dtype.name not in undefined:
            undefined += [t.dtype.name]

    #Add variables that appear in kernel, but not in kernel.variables, to undefined
    for w in FindVariables(unique=True).visit(kernel.body):
        if w.name not in kernel.variables and w.name not in undefined:
            undefined += [w.name]

    #Generate the imports in driver missing from kernel
    new_imports = []
    for i in FindNodes(Import).visit(driver.spec):

        #List variables appearing in imports and undefined
        new_s = []
        for s in i.symbols:
            if s.name in undefined:
                new_s += [s.clone()]

        #If import has any undefined symbols, add new import with only undefined symbols
        if(new_s):
            new_imports += [i.clone(symbols=as_tuple(new_s))]

    #Add any new imports at beginning of kernel spec
    for i in new_imports:
        kernel.spec.prepend(i)


def hoist_fun(driver):

    # Declare dimension objects needed for transformation
    horizontal = Dimension(name='horizontal', size='klon', index='jlon', bounds=('kidia', 'kfdia'))
    vertical = Dimension(name='vertical', size='klev+1', index='jlev', bounds=('0', 'klev'))
    gdim = Dimension(name='gdim', size='3', index='jg', bounds=('1', '3'))

    vertical1 = Dimension(name='vertical1', size='klev+1', index='jlev1', bounds=('0', 'klev'))
    vertical2 = Dimension(name='vertical2', size='klev+1', index='jlev2', bounds=('0', 'klev'))

    #Create a module object
    mod = make_module(driver)

    #Remove associate statements from source code
    remove_associate(driver)

    insert_routine_body(driver)

    #Kernels are the subroutines contained in driver in this case.
    kernels = list(driver.members)

    remove_hook(driver)

#    for kernel in kernels:
#
#        remove_hook(kernel)
#
#        pass_undefined(driver, kernel)
#
#        insert_imports(driver, kernel)
#
#        hoist_loops(driver, kernel, horizontal)
#
#        remove_index_arg(driver, kernel, 'KL')
#
#        remove_unused_variables(kernel)
#
#        remove_unused_arguments(driver, kernel)
#
#        pass_value(kernel)
#
#        parametrize(kernel)
#
#        kernel.parent = None
#
#    reorder_arrays(driver, kernels)
#
    reorder_arrays_dim(driver, horizontal)

    move_independent_loop_out(horizontal, driver)

    parametrize(driver)
#
#    add_seq(kernels)
#
    add_acc(driver, vector=horizontal, sequential=vertical)

    add_data(driver)

    remove_unused_variables(driver)

    mod = mod.clone(contains = Section(body= (driver.clone(contains=None),) + tuple(kernels)))

    return mod


def hoist(argv):

    original_name = Path(argv[1]).resolve()

    if len(argv) == 2:
        new_name = original_name.parent / ('new_' + original_name.name)
    else:
        new_name = Path(argv[2]).resolve() / ('new_' + original_name.name)

    sauce_file = Sourcefile.from_file(original_name)

    # Driver is the first subroutine in the file
    driver = sauce_file.subroutines[0]

    start = time()
    mod = hoist_fun(driver)
    end = time()
    print('Hoist time:', end - start)

    new_file = Sourcefile(path=new_name, ir = Section(body = mod))

    new_file.write()



if __name__ == "__main__":
    hoist(argv)

