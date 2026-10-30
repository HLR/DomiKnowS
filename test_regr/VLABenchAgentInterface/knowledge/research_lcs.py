"""One declarative source for research admission, verification and token masks."""
from domiknows.generation.lc_relay import RelayedKnowledge


def declare_research(rules):
    from domiknows.graph import Concept
    from domiknows.graph.logicalConstrain import andL, notL, existsL
    row=Concept(name="research_requirement")
    fields={name:row(name=name) for name in (
        "known","holds","freshness","observed","current","dependencies_valid",
        "spatial","temporal","geometry","grounding","complete","robust","joint","abstain")}
    hypothesis=Concept(name="research_hypothesis")
    alive=hypothesis(name="alive");compatible=hypothesis(name="compatible")
    different=Concept(name="research_different_binding")
    different.has_a(left=hypothesis,right=hypothesis)
    def rule(name,kind,consequent):
        rules.rule("research__"+name,"affordance",lambda:fields[kind]("r"),consequent)
    rule("fresh_observed","freshness",lambda:andL(fields["known"]("r"),fields["observed"]("r")))
    for name in ("current","dependencies_valid"):
        rules.rule("research__fresh_"+("current" if name=="current" else "dependencies"),"affordance",
            lambda:andL(fields["freshness"]("r"),fields["known"]("r")),lambda n=name:fields[n]("r"))
    for kind in ("spatial","temporal","geometry"):
        rule(kind+"_known",kind,lambda:fields["known"]("r"))
        rules.rule("research__"+kind+"_holds","affordance",lambda k=kind:andL(fields[k]("r"),fields["known"]("r")),lambda:fields["holds"]("r"))
    rules.rule("research__grounding_complete","affordance",
        lambda:andL(fields["grounding"]("r"),notL(fields["joint"]("r"))),lambda:fields["complete"]("r"))
    rules.rule("research__grounding_robust","affordance",
        lambda:andL(fields["grounding"]("r"),fields["robust"]("r"),hypothesis("h"),alive("h")),
        lambda:compatible("h"))
    rules.rule("research__grounding_joint","affordance",
        lambda:andL(fields["grounding"]("r"),fields["joint"]("r")),
        lambda:existsL(andL(hypothesis("h"),alive("h"),compatible("h"))))
    rules.rule("research__grounding_unique","affordance",
        lambda:andL(fields["grounding"]("r"),fields["abstain"]("r")),
        lambda:notL(existsL(andL(hypothesis("h"),alive("h"),hypothesis("j"),alive("j"),different("h","j")))))
    rules.rule("research__grounding_abstain_match","affordance",
        lambda:andL(fields["grounding"]("r"),fields["abstain"]("r")),
        lambda:existsL(andL(hypothesis("h"),alive("h"),compatible("h"))))
    rule("grounding_available","grounding",lambda:andL(fields["known"]("r"),existsL(hypothesis("h"))))
    return ([(c,n,"research","immediate") for n,c in fields.items()]
            +[(hypothesis,"hypothesis","research","immediate"),(alive,"alive","research","immediate"),
              (compatible,"compatible","research","immediate"),(different,"different","research","immediate")])


def build_research_knowledge(active=True):
    return RelayedKnowledge("vlabench_research",declare_research,active=active)
