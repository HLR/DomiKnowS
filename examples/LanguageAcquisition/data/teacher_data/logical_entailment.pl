(:- set_prolog_flag(stack_limit, limit(100000000000)).)
(cls :- write('\33\[2J').)

(:- discontiguous(st1/1).)
(:- discontiguous(he1/1).)
(:- discontiguous(el1/1).)
(:- discontiguous(ci1/1).)
(:- discontiguous(sq1/1).)
(:- discontiguous(tr1/1).)
(:- discontiguous(re1/1).)
(:- discontiguous(bl1/1).)
(:- discontiguous(gr1/1).)
(:- discontiguous(or1/1).)
(:- discontiguous(pu1/1).)
(:- discontiguous(ye1/1).)
(:- discontiguous(bi1/1).)
(:- discontiguous(sm1/1).)
(:- discontiguous(me1/1).)

%size
:- dynamic(bi1/1).
:- dynamic(sm1/1).
:- dynamic(me1/1).

%color
:- dynamic(re1/1).
:- dynamic(bl1/1).
:- dynamic(ye1/1).
:- dynamic(or1/1).
:- dynamic(gr1/1).
:- dynamic(pu1/1).

%shape
:- dynamic(st1/1).
:- dynamic(tr1/1).
:- dynamic(he1/1).
:- dynamic(sq1/1).
:- dynamic(ci1/1).
:- dynamic(el1/1).

%position
:- dynamic(le2/2).
:- dynamic(ab2/2).


% This script contains the rules needed to determine if a meaning denotes a situation.

% This rule evalautes the predicate sequences P(X), where X is a variable that will be unified with the atoms present in the given situation. If the mapping between variables and constants is unique, we can say that the given predicate sequence denotes the situation.

mapping([]).
(mapping([P|T]) :- catch((current_predicate(_,P),P), error(_,_Context), !),!,mapping(T).)

(entails(S) :- catch((term_string(P,S),P), error(_,_Context), false).)

% This rule determines if the predicates in the meaning are present in the situation, regardless of the variables and constants. 
%(find_predicates([]):- true.)

%(find_predicates([P|T]) :- current_predicate(_,P), find_predicates(T).)

(find_predicates(List,Length) :- find_predicates(List,[],Length).)

find_predicates([],Length,Length).

(find_predicates([P|Tail],Accumulator,Length) :- (current_predicate(_,P),call(P) -> term_string(P,S),find_predicates(Tail,[S|Accumulator],Length); find_predicates(Tail,Accumulator,Length)).)

