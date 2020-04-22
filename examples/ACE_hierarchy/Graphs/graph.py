from regr.graph import Concept, Relation, Graph
# from .base import NewGraph as Graph
from regr.graph.logicalConstrain import *

Graph.clear()
Concept.clear()
Relation.clear()

with Graph('global') as graph:
    graph.ontology = ('http://ontology.ihmc.us/ML/ACE.owl', './')

    with Graph('linguistic') as ling_graph:
        word = Concept('word')
        phrase = Concept('phrase')
        sentence = Concept('sentence')
        (rel_sentence_contains_phrase,) = sentence.contains(phrase)
        (rel_sentence_contains_word,) = sentence.contains(word)

    with Graph('application') as app_graph:
        FAC = word(name='FAC')
        GPE = word(name='GPE')
        LOC = word(name='LOC')
        ORG = word(name='ORG')
        PER = word(name='PER')
        VEH = word(name='VEH')
        WEA = word(name='WEA')

        FAC.not_a(GPE, LOC, VEH, WEA, PER, ORG)

        GPE.not_a(FAC, LOC, VEH, WEA, PER, ORG)

        LOC.not_a(FAC, GPE, VEH, WEA, PER, ORG)

        VEH.not_a(FAC, GPE, LOC, WEA, PER, ORG)

        WEA.not_a(FAC, GPE, LOC, VEH, PER, ORG)

        PER.not_a(FAC, GPE, LOC, VEH, WEA, ORG)

        ORG.not_a(FAC, GPE, LOC, VEH, WEA, PER)


        #FAC SUB
        Airport = FAC(name="Airport")
        Building_Grounds = FAC(name="Building-Grounds")
        Path = FAC(name="Path")
        Plant = FAC(name="Plant")
        Subarea_Facility = FAC(name="Subarea-Facility")

        Airport.not_a(Building_Grounds, Path, Plant, Subarea_Facility)
        Building_Grounds.not_a(Airport, Path, Plant, Subarea_Facility)
        Path.not_a(Building_Grounds, Airport, Plant, Subarea_Facility)
        Plant.not_a(Building_Grounds, Path, Airport, Subarea_Facility)
        Subarea_Facility.not_a(Building_Grounds, Path, Plant, Airport)

        #GPE SUB
        Continent = GPE(name="Continent")
        County_or_District = GPE(name="County-or-District")
        GPE_Cluster = GPE(name="GPE-Cluster")
        Nation = GPE(name="Nation")
        Population_Center = GPE(name="Population-Center")
        Special = GPE(name="Special")
        State_or_Province = GPE(name="State-or-Province")

        Continent.not_a(County_or_District, GPE_Cluster, Nation, Population_Center, Special, State_or_Province)
        County_or_District.not_a(Continent, GPE_Cluster, Nation, Population_Center, Special, State_or_Province)
        GPE_Cluster.not_a(County_or_District, Continent, Nation, Population_Center, Special, State_or_Province)
        Nation.not_a(County_or_District, GPE_Cluster, Continent, Population_Center, Special, State_or_Province)
        Population_Center.not_a(County_or_District, GPE_Cluster, Nation, Continent, Special, State_or_Province)
        Special.not_a(County_or_District, GPE_Cluster, Nation, Population_Center, Continent, State_or_Province)
        State_or_Province.not_a(County_or_District, GPE_Cluster, Nation, Population_Center, Special, Continent)

        #LOC SUB
        Address = LOC(name="Address")
        Boundary = LOC(name="Boundary")
        Celestial = LOC(name="Celestial")
        Land_Region_Natural = LOC(name="Land-Region-Natural")
        Region_General = LOC(name="Region-General")
        Region_International = LOC(name="Region-International")
        Water_Body = LOC(name="Water-Body")

        Address.not_a(Boundary, Celestial, Land_Region_Natural, Region_General, Region_International, Water_Body)
        Boundary.not_a(Address, Celestial, Land_Region_Natural, Region_General, Region_International, Water_Body)
        Celestial.not_a(Address, Boundary, Land_Region_Natural, Region_General, Region_International, Water_Body)
        Land_Region_Natural.not_a(Address, Boundary, Celestial, Region_General, Region_International, Water_Body)
        Region_General.not_a(Address, Boundary, Celestial, Land_Region_Natural, Region_International, Water_Body)
        Region_International.not_a(Address, Boundary, Celestial, Land_Region_Natural, Region_General, Water_Body)
        Water_Body.not_a(Address, Boundary, Celestial, Land_Region_Natural, Region_General, Region_International)

        #ORG SUB
        Commercial = ORG(name="Commercial")
        Educational = ORG(name="Educational")
        Entertainment = ORG(name="Entertainment")
        Government = ORG(name="Government")
        Media = ORG(name="Media")
        Medical_Science = ORG(name="Medical-Science")
        Non_Governmental = ORG(name="Non-Governmental")
        Religious = ORG(name="Religious")
        Sports = ORG(name="Sports")

        Commercial.not_a(Educational, Entertainment, Government, Media, Medical_Science, Non_Governmental, Religious, Sports)
        Educational.not_a(Commercial, Entertainment, Government, Media, Medical_Science, Non_Governmental, Religious, Sports)
        Entertainment.not_a(Commercial, Educational, Government, Media, Medical_Science, Non_Governmental, Religious, Sports)
        Government.not_a(Commercial, Educational, Entertainment, Media, Medical_Science, Non_Governmental, Religious, Sports)
        Media.not_a(Commercial, Educational, Entertainment, Government, Medical_Science, Non_Governmental, Religious, Sports)
        Medical_Science.not_a(Commercial, Educational, Entertainment, Government, Media, Non_Governmental, Religious, Sports)
        Non_Governmental.not_a(Commercial, Educational, Entertainment, Government, Media, Medical_Science, Religious, Sports)
        Religious.not_a(Commercial, Educational, Entertainment, Government, Media, Medical_Science, Non_Governmental, Sports)
        Sports.not_a(Commercial, Educational, Entertainment, Government, Media, Medical_Science, Non_Governmental, Religious)

        #PER SUB
        Group = PER(name="Group")
        Indeterminate = PER(name="Indeterminate")
        Individual = PER(name="Individual")

        Group.not_a(Indeterminate, Individual)
        Indeterminate.not_a(Group, Individual)
        Individual.not_a(Group, Indeterminate)

        #VEH SUB
        Air = VEH(name="Air")
        Land = VEH(name="Land")
        Subarea_Vehicle = VEH(name="Subarea-Vehicle")
        Underspecified = VEH(name="Underspecified")
        Water = VEH(name="Water")

        Air.not_a(Land, Subarea_Vehicle, Underspecified, Water)
        Land.not_a(Air, Subarea_Vehicle, Underspecified, Water)
        Subarea_Vehicle.not_a(Air, Land, Underspecified, Water)
        Underspecified.not_a(Air, Land, Subarea_Vehicle, Water)
        Water.not_a(Air, Land, Subarea_Vehicle, Underspecified)

        #WEA SUB
        Biological = WEA(name="Biological")
        Blunt = WEA(name="Blunt")
        Chemical = WEA(name="Chemical")
        Exploding = WEA(name="Exploding")
        Nuclear = WEA(name="Nuclear")
        Projectile = WEA(name="Projectile")
        Sharp = WEA(name="Sharp")
        Shooting = WEA(name="Shooting")
        WEA_Underspecified = WEA(name="WEA-Underspecified")

        Biological.not_a(Blunt, Chemical, Exploding, Nuclear, Projectile, Sharp, Shooting, WEA_Underspecified)
        Blunt.not_a(Biological, Chemical, Exploding, Nuclear, Projectile, Sharp, Shooting, WEA_Underspecified)
        Chemical.not_a(Biological, Blunt, Exploding, Nuclear, Projectile, Sharp, Shooting, WEA_Underspecified)
        Exploding.not_a(Biological, Blunt, Chemical, Nuclear, Projectile, Sharp, Shooting, WEA_Underspecified)
        Nuclear.not_a(Biological, Blunt, Chemical, Exploding, Projectile, Sharp, Shooting, WEA_Underspecified)
        Projectile.not_a(Biological, Blunt, Chemical, Exploding, Nuclear, Sharp, Shooting, WEA_Underspecified)
        Sharp.not_a(Biological, Blunt, Chemical, Exploding, Nuclear, Projectile, Shooting, WEA_Underspecified)
        Shooting.not_a(Biological, Blunt, Chemical, Exploding, Nuclear, Projectile, Sharp, WEA_Underspecified)
        WEA_Underspecified.not_a(Biological, Blunt, Chemical, Exploding, Nuclear, Projectile, Sharp, Shooting)