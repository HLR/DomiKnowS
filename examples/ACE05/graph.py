from regr.graph import Graph, Concept, Relation
from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import ifL, orL, andL


Graph.clear()
Concept.clear()
Relation.clear()


with Graph('global') as graph:
    with Graph('linguistic') as ling_graph:
        word = Concept(name='word')
        phrase = Concept(name='phrase')
        sentence = Concept(name='sentence')
        phrase.has_many(word)
        sentence.has_many(phrase)

        pair = Concept(name='pair')
        pair.has_a(phrase, phrase)

    with Graph('ACE05') as ace05:
        with Graph('Entities') as entities_graph:
            # ACE (Automatic Content Extraction) English Annotation Guidelines for Entities
            # Version 5.6.6 2006.08.01

            # Entity - Entities may be referenced in a text by their name, indicated by a common noun or noun phrase, or represented by a pronoun.
            entity = phrase(name='Entity')

            # Entity Class - Each taggable entity must be assigned a class that describes the kind of reference the entity makes to something in the world.
            # TODO: this is a property has a value being one of NEG, SPC, GEN, USP
            # TODO: "must be assigned a class" suggest a constraint that a class is required
            entity['Class']
            # An alternative way is creating subtypes for classes and mention disjoint among them
            # Negatively Quantified (NEG) - An entity is NEG when it has been quantified such that it refers to the empty set of the type of object mentioned.
            neg = entity(name='Class.NEG')
            # Specific Referential (SPC) - An entity is SPC when the entity being referred to is a particular, unique object (or set of objects), whether or not the author or reader is aware of the name of the entity or its anchor in the (local) real world.
            spc = entity(name='Class.SPC')
            # Generic Referential (GEN) - An entity is GEN when the entity being referred to is not a particular, unique object (or set of objects). Instead GEN entities refer to a kind or type of entity.
            gen = entity(name='Class.GEN')
            # Under-specified Referential (USP) - We reserve the term underspecified for non-generic non-specific reference.
            usp = entity(name='Class.USP')

            # disjoint
            disjoint(neg, spc, gen, usp)

            # Types
            # Person - Person entities are limited to humans. A person may be a single individual or a group
            person = entity(name='Person')
            # Organization - Organization entities are limited to corporations, agencies, and other groups of people defined by an established organizational structure.
            organization = entity(name='Organization')
            # GPE (Geo-political Entity) - GPE entities are geographical regions defined by political and/or social groups. A GPE entity subsumes and does not distinguish between a nation, its region, its government, or its people.
            gpe = entity(name='GPE')
            # Location - Location entities are limited to geographical entities such as geographical areas and landmasses, bodies of water, and geological formations.
            location = entity(name='Location')
            # Facility - Facility entities are limited to buildings and other permanent man-made structures and real estate improvements.
            facility = entity(name='Facility')
            # Vehicle – A vehicle entity is a physical device primarily designed to move an object from one location to another, by (for example) carrying, pulling, or pushing the transported object. Vehicle entities may or may not have their own power source.
            vehicle = entity(name='Vehicle')
            # Weapon – Weapon entities are limited to physical devices primarily used as instruments for physically harming or destroying other entities.
            weapon = entity(name='Weapon')

            # disjoint
            disjoint(person, organization, gpe, location, facility, vehicle, weapon)

            # Abbreviation
            PER = person
            ORG = organization
            GPE = gpe
            LOC = location
            FAC = facility
            VEH = vehicle
            WEA = weapon

            # person subtypes
            # PER.Individual - If the Person entity refers to a single person, tag it as PER.Individual.
            individual = person('Individual')
            # PER.Group - If the Person entity refers to more than one person, tag it as PER.Group unless the group meets the requirements of an Organization or a GPE described below. This will include family names and ethnic and religious groups that do not have a formal organization unifying them.
            group = person('Group')
            # PER.Indefinite - If from the context you can’t judge whether the Person entity refers to one or more than one person, tag it as PER.Indefinite.
            indefinite = person('Indefinite')
            # disjoint
            disjoint(individual, group, indefinite)

            # organization subtypes
            # Government (GOV) - Government organizations are those that are of, relating to, or dealing with the structure or affairs of government, politics, or the state.
            government = organization('Government')
            # Commercial (COM) - A commercial organization is an entire organization or a taggable portion of an organization that is focused primarily upon providing ideas, products, or services for profit.
            commercial = organization('Commercial')
            # Educational (EDU) - An educational organization is an entire institution or taggable portion of an institution that is focused primarily upon the furthering or promulgation of learning/education.
            educational = organization('Educational')
            # Entertainment (ENT) - Entertainment organizations are those whose primary activity is entertainment.
            entertainment = organization('Entertainment')
            # Non-Governmental Organizations (NonGov) - Non-governmental organizations are those organizations that are not a part of a government or commercial organization and whose main role is advocacy, charity or politics (in a broad sense).
            non_governmental = organization('Non-Governmental')
            # Media (MED) - Media organizations are those whose primary interest is the distribution of news or publications, regardless of whether the organization is privately or publicly owned.
            media = organization('Media')
            # Religious (REL) - Religious organizations are those that are primarily devoted to issues of religious worship.
            religious = organization('Religious')
            # Medical-Science (SCI) - Medical-Science organizations are those whose primary activity is the application of medical care or the pursuit of scientific research, regardless of whether that organization is publicly or privately owned.
            medical_science = organization('Medical-Science')
            # Sports (SPO) - Sports organizations are those that are primarily concerned with participating in or governing organized sporting events, whether professional, amateur, or scholastic.
            sports = organization('Sports')
            # NOTE: The collection of organization subtypes is hierarchical in nature. ACE05 assigns the most specific type possible. So the annotated results are disjoint.
            # disjoint
            disjoint(government, commercial, educational, entertainment, non_governmental, media, religious, medical_science, sports)

            # GPE subtypes
            # Continent - Taggable mentions of the entireties of any of the seven continents.
            continent = organization('Continent')
            # Nation - Taggable mentions of the entireties of any nation.
            nation = organization('Nation')
            # State-or-Province - Taggable mentions of the entireties of any state, province, or canton of any nation.
            state_or_province = organization('State-or-Province')
            # County-or-District - Taggable mentions of the entireties of any county, district, prefecture, or analogous body of any state/province/canton.
            county_or_district = organization('County-or-District')
            # Population-Center - Taggable mentions of the entireties of any GPE below the level of County-or- District.
            population_center = organization('Population-Center')
            # GPE-Cluster - Named groupings of GPEs that can function as political entities.
            gpe_cluster = organization('GPE-Cluster')
            # Special - A closed set of GPEs for which the conventional labels do not straightforwardly apply.
            special = organization('Special')
            # disjoint
            disjoint(continent, nation, state_or_province, county_or_district, population_center, gpe_cluster, special)

            # GPE Roles - Annotators need to decide for each entity mention in the text which role (Person, Organization, Location, GPE) the context of that mention invokes. This judgment typically depends on the relations that the entity enters into.
            # TODO: Similar to Class, this is a field required from one the the following four.
            gpe['Role']
            # For now, implemented as disjoint subtypes
            # GPE.ORG - GPE.ORG is used for GPE mentions that refer to the entire governing body of a GPE.
            gpe_org = gpe('Role.ORG')
            # GPE.PER - Populations of a GPE are treated as GPE.PER
            gpe_per = gpe('Role.PER')
            # GPE.LOC - GPE.LOC is used when a mention of a GPE entity primarily references the territory or geographic position of the GPE.
            gpe_loc = gpe('Role.LOC')
            # GPE.GPE - GPE.GPE is used when more than one of the other GPE roles is being referenced at once or when no one role stands out in the context.
            gpe_gpe = gpe('Role.GPE')
            # disjoint
            disjoint(gpe_org, gpe_per, gpe_loc, gpe_gpe)

            # Location subtypes
            # Address - A location denoted as a point such as in a postal system or abstract coordinates. The name of a location in a postal system is also an address.
            address = location('Address')
            # Boundary - A one-dimensional location such as a border between GPE’s or other locations.
            boundary = location('Boundary')
            # Celestial - A location which is otherworldly or entire-world-inclusive.
            celestial = location('Celestial')
            # Water-Body - Bodies of water, natural or artificial (man-made).
            water_body = location('Water-Body')
            # Land-Region-natural - Geologically or ecosystemically designated, non-artificial locations.
            land_region_natural = location('Land-Region-natural')
            # Region-International - Taggable locations that cross national borders.
            region_international = location('Region-International')
            # Region-General - Taggable locations that do not cross national borders.
            region_general = location('Region-General')
            # disjoint
            disjoint(address, boundary, celestial, water_body, land_region_natural, region_international, region_general)

            # Facilities subtypes
            # Airport - A facility whose primary use is as an airport.
            airport = facility('Airport')
            # Plant - One or more buildings that are used and/or designed solely for industrial purposes: manufacturing, power generation, etc.
            plant = facility('Plant')
            # Building-or-Grounds - Man-made/-maintained buildings, outdoor spaces, and other such facilities.
            building_or_grounds = facility('Building-or-Grounds')
            # Subarea-Facility - Taggable portions of facilities. The threshold of taggability of subarea-facility is the ability of the area to contain a normally proportioned person comfortably.
            subarea_facility = facility('Subarea-Facility')
            # Path - A facility that allows fluids, energies, persons or vehicles to pass from one location to another.
            path = facility('Path')
            # disjoint
            disjoint(airport, plant, building_or_grounds, subarea_facility, path)

            # Vehicle subtypes
            # Air - Vehicles designed to locomote primarily through the air, not touching water or land.
            air = vehicle('Air')
            # Land - Vehicles designed to locomote primarily upon land.
            land = vehicle('Land')
            # Water - Vehicles designed to locomote primarily on or submerged in water.
            water = vehicle('Water')
            # Subarea-Vehicle - A portion of a Vehicle entity that is of a size such that humans can fit inside with some degree of comfort.
            subarea_vehicle = vehicle('Subarea-Vehicle')
            # Underspecified - Vehicles whose subtype is not specified in the text, or sets of vehicles of different subtypes.
            underspecified_vihicle = vehicle('Underspecified-Vehicle')

            # Weapon subtypes
            # Blunt - Blunt weapons are those designed or used as bludgeoning instruments.
            blunt = vehicle('Blunt')
            # Exploding - Exploding weapons are those that are designed or used to accomplish damage through explosion.
            exploding = vehicle('Exploding')
            # Sharp - Sharp weapons are those designed or used to cut, slash, jab, & hack.
            sharp = vehicle('Sharp')
            # Chemical - A chemical weapon is any device or substance that is designed or has been used for the purpose of causing death or serious injury through the release, dissemination or impact of toxic or poisonous chemicals or their immediate precursors.
            chemical = vehicle('Chemical')
            # Biological - Biological weapons are bacteria, viruses, fungi, toxins, as well as the means of their dispersal, used for the spread of disease among humans, plants & animals.
            biological = vehicle('Biological')
            # Shooting - Shooting weapons are weapons designed or used to send projectile objects at great speed for the purpose of causing harm.
            shooting = vehicle('Shooting')
            # Projectile - Projectile weapons are weapons designed or used to be projected at great speed for the purpose of causing harm.
            projectile = vehicle('Projectile')
            # Nuclear - Nuclear weapons are those designed or used for the purpose of causing damage, death, and harm through the expenditure of radiological or nuclear energies.
            nuclear = vehicle('Nuclear')
            # Underspecified - Underspecified weapons are weapons whose subtype is not specified in the text, or sets of weapons of different subtypes.
            underspecified_weapon = vehicle('Underspecified-Weapon')

        with Graph('Relations') as relations_graph:
            # ACE (Automatic Content Extraction) English Annotation Guidelines for Relations
            # Version 5.8.3 – 2005.07.01

            # Relation - Relations are ordered pairs of entities.
            relation = pair(name='Relation')
            relation.has_a(arg1=entity, arg2=entity)

            # Modality
            relation['Modality']
            # Asserted - when the Reasonable Reader Rule is interpreted relative to the 'Real' world;
            # Other - when the Reasonable Reader Rule is taken to hold in a particular counterfactual world.
            
            # Tense - TENSE will be defined relative to the time of speech.
            relation['Tense']
            # Past - the Relation is taken to hold only for some span prior to the time of speech;
            # Future - the Relation is taken to hold only for some span after the time of speech;
            # Present - the Relation is taken to hold for a limited time overlapping with the time of speech;
            # Unspecified - the Relation is ‘static’ or the span of time for which it holds cannot be determined with certainty;

            # Classes
            relation['Class']
            # Possessive - The Possessive Syntactic class is used when the Entity Mention of one argument is possessive case and the Entity Mention of the other argument is clearly the ‘possessed object’ in the construction.
            # Preposition - The Preposition Syntactic class is used when the one entity mention is linked to the other with a Preposition.
            # PreMod - The PreMod Syntactic Class is used for those Relations that are established by the construction in which a proper adjective or proper noun modifies a taggable entity.
            # Coordination - The Coordination Syntactic Class is used for Relations that are expressed using noun phrases containing the coordinating conjunction ‘and’.
            # Formulaic - There are a number of constructions that are commonly used in news stories.
            # Participial - The Syntactic Class Participial will be used in cases where there is a taggable Relation between a head noun and a noun contained within a participial phrase that modifies it.
            # Verbal - The Syntactic Class Verbal will be used for cases motivated by a taggable mention of a Relation between two entities where the Relation is directly expressed by a verb tying the two together into a sentence or a clause.
            # Other - The Other Class of Relations is reserved for those that do not strictly satisfy the syntactic requirements of one of the other classes, but still satisfies the ‘beyond a reasonable doubt’ standard for Relation taggability.

            # TODO: there are Timestamping for relation

            # Types
            # Physical
            physical = relation(name='Physical')
            # Physical.Located - The Located Relation captures the physical location of an entity.
            located = physical(name='Located')
            located.has_a(arg1=person, arg2=entity)
            # TODO: arg2 is one of FAC, LOC, GPE
            ifL(located, ('x', 'y'), orL(FAC, 'y', orL(LOC, 'y', GPE, 'y')))
            # Physical.Near - Near indicates that an entity is explicitly near another entity, but neither entity is a part of the other or located in/at the other.
            near = physical(name='Near')
            near.has_a(arg1=entity, arg2=entity)
            # TODO: arg1 is one of PER, FAC, GPE, LOC
            ifL(near, ('x', 'y'), orL(PER, 'x', orL(FAC, 'x', orL(GPE, 'x', LOC, 'x'))))
            # TODO: arg2 is one of FAC, GPE, LOC
            ifL(near, ('x', 'y'), orL(FAC, 'y', orL(GPE, 'y', LOC, 'y')))

            # Part-whole
            part_whole = relation(name='Part-whole')
            # Part-whole.Geographical - The Geographical Relation captures the location of a Facility, Location, or GPE in or at or as a part of another Facility, Location, or GPE.
            geographical = part_whole(name='Geographical')
            geographical.has_a(arg1=entity, arg2=entity)
            # TODO: arg1 is one of FAC, LOC, GPE
            ifL(geographical, ('x', 'y'), orL(FAC, 'x', orL(LOC, 'x', GPE, 'x')))
            # TODO: arg2 is one of FAC, LOC, GPE
            ifL(geographical, ('x', 'y'), orL(FAC, 'y', orL(LOC, 'y', GPE, 'y')))
            # Part-whole.Subsidiary - Subsidiary captures the ownership, administrative, and other hierarchical relationships between organizations and between organizations and GPEs.
            subsidiary = part_whole(name='Subsidiary')
            subsidiary.has_a(arg1=organization, arg2=entity)
            # TODO: arg2 is one of ORG, GPE
            ifL(subsidiary, ('x', 'y'), orL(ORG, 'y', GPE, 'y'))
            # Part-whole.Artifact - Artifact characterizes physical relationships between concrete physical objects and their parts.
            artifact = part_whole(name='Artifact')
            artifact.has_a(arg1=entity, arg2=entity)
            # TODO: (arg1 is VEH and arg2 is VEH) or (arg1 is WEA and arg2 is WEA)
            ifL(artifact, ('x', 'y'), orL(andL(VEH, 'x', VEH, 'y'), andL(WEA, 'x', WEA, 'y')))

            # Personal-Social - Personal-Social relations describe the relationship between people. Both arguments must be entities of type PER.
            # The arguments of these Relations are not ordered. The Relations are symmetric.
            personal_social = relation(name='Personal-Social')
            personal_social.has_a(arg1=person, arg2=person)
            # Personal-Social.Business - The Business Relation captures the connection between two entities in any professional relationship.
            business = personal_social(name='Business')
            # Personal-Social.Family - The Family Relation captures the connection between one entity and another with which it is in any familial relationship.
            family = personal_social(name='Family')
            # Personal-Social.Lasting-Personal - The relationship must involve personal contact (or a reasonable assumption thereof); and there must be some indication or expectation that the relationship exists outside of a particular cited interaction.
            lasting_personal = personal_social(name='Lasting-Personal')
            
            # ORG-Affiliation
            org_affiliation = relation(name='ORG-Affiliation')
            # ORG-Affiliation.Employment - Employment captures the relationship between Persons and their employers.
            employment = org_affiliation(name='Employment')
            employment.has_a(arg1=person, arg2=entity)
            # TODO: arg2 is one of ORG, GPE
            ifL(employment, ('x', 'y'), orL(ORG, 'y', GPE, 'y'))
            # ORG-Affiliation.Ownership - Ownership captures the relationship between a Person and an Organization owned by that Person.
            ownership = org_affiliation(name='Ownership')
            ownership.has_a(arg1=person, arg2=organization)
            # ORG-Affiliation.Founder - Founder captures the relationship between an agent (Person, Organization, or GPE) and an Organization or GPE established or set up by that agent.
            founder = org_affiliation(name='Founder')
            founder.has_a(arg1=entity, arg2=entity)
            # TODO: arg1 is one of PER, ORG
            ifL(founder, ('x', 'y'), orL(PER, 'x', ORG, 'x'))
            # TODO: arg2 is one of ORG, GPE
            ifL(founder, ('x', 'y'), orL(ORG, 'y', GPE, 'y'))
            # ORG-Affiliation.Student-Alum - Student-Alum captures the relationship between a Person and an educational institution the Person attends or attended.
            student_alum = org_affiliation(name='Student-Alum')
            student_alum.has_a(arg1=person, arg2=educational)
            # ORG-Affiliation.Sports-Affiliation - Sports-Affiliation captures the relationship between a player, coach, manager, or assistant and his or her affiliation with a sports organization.
            sports_affiliation = org_affiliation(name='Sports-Affiliation')
            sports_affiliation.has_a(arg1=person, arg2=organization)
            # ORG-Affiliation.Investor-Shareholder - Investor-Shareholder captures the relationship between an agent (Person, Organization, or GPE) and an Organization in which the agent has invested or in which the agent owns shares/stock.
            investor_shareholder = org_affiliation(name='Investor-Shareholder')
            investor_shareholder.has_a(arg1=entity, arg2=entity)
            # TODO: arg1 is one of PER, ORG, GPE
            ifL(investor_shareholder, ('x', 'y'), orL(PER, 'x', orL(ORG, 'x', GPE, 'x')))
            # TODO: arg2 is one of ORG, GPE
            ifL(investor_shareholder, ('x', 'y'), orL(ORG, 'y', GPE, 'y'))
            # ORG-Affiliation.Membership - Membership captures the relationship between an agent and an organization of which the agent is a member.
            membership = org_affiliation(name='Membership')
            membership.has_a(arg1=entity, arg2=organization)
            # TODO: arg1 is one of PER, ORG, GPE
            ifL(membership, ('x', 'y'), orL(PER, 'x', orL(ORG, 'x', GPE, 'x')))

            # Agent-Artifact
            agent_artifact = relation('Agent-Artifact')
            # Agent-Artifact.User-Owner-Inventor-Manufacturer - This Relation applies when an agent owns an artifact, has possession of an artifact, uses an artifact, or caused an artifact to come into being.
            user_owner_inventor_manufacturer = agent_artifact(name='User-Owner-Inventor-Manufacturer')
            user_owner_inventor_manufacturer.has_a(arg1=entity, arg2=entity)
            # TODO: arg1 is one of PER, ORG, GPE
            ifL(user_owner_inventor_manufacturer, ('x', 'y'), orL(PER, 'x', orL(ORG, 'x', GPE, 'x')))
            # TODO: arg2 is one of WEA, VEH, FAC
            ifL(user_owner_inventor_manufacturer, ('x', 'y'), orL(WEA, 'y', orL(VEH, 'y', FAC, 'y')))
            
            # Gen-Affiliation
            gen_affiliation = relation('Gen-Affiliation')
            # Gen-Affiliation.Citizen-Resident-Religion-Ethnicity - Citizen-Resident-Religion-Ethnicity describes the Relation between a PER entity and PER.Group, LOC, GPE, ORG
            citizen_resident_religion_ethnicity = gen_affiliation('Citizen-Resident-Religion-Ethnicity')
            citizen_resident_religion_ethnicity.has_a(arg1=person, arg2=entity)
            # TODO: arg2 is one of PER.Group, LOC, GPE, ORG
            ifL(user_owner_inventor_manufacturer, ('x', 'y'), orL(group, 'y', orL(LOC, 'y', orL(GPE, 'y', ORG, 'y'))))
            # Gen-Affiliation.Org-Location-Origin - Org-Location-Origin captures the relationship between an organization and the LOC or GPE where it is located, based, or does business.
            org_location_origin = gen_affiliation('Org-Location-Origin')
            org_location_origin.has_a(arg1=organization, arg2=entity)
            # TODO: arg2 is one of LOC, GPE
            ifL(org_location_origin, ('x', 'y'), orL(LOC, 'y', GPE, 'y'))

        with Graph('Events') as events_graph:
            # ACE (Automatic Content Extraction) English Annotation Guidelines for Events
            # Version 5.4.3 2005.07.01

            trigger = word(name='trigger')
            # NOTE: do we need the abstract event or base it on trigger?
            event = Concept(name='Event')
            event.has_a(trigger)
            participant = entity(name='Participant')
            attribute = entity(name='Attribute')

            # NOTE: there can be variable number of participant(s) and attribute(s)
            # should we create a new Relation or concept based on pair?
            #
            # Relation:
            @Concept.relation_type('involve')
            class Involve(Relation): pass
            event.involve(participant)
            event.with_(attribute)
            #
            # or Concept:
            # involve = pair(name='Involve')
            # involve.has_a(event, participant)
            # with_ = pair(name='With')
            # with_.has_a(event, attribute)

            # Polarity [POSITIVE, NEGATIVE]- An Event is NEGATIVE when it is explicitly indicated that the Event did not occur (see examples). All other Events are POSITIVE.
            event['Polarity']
            # Tense [PAST, FUTURE, PRESENT] - TENSE is determined with respect to the speaker or author.
            event['Tense']
            # Genericity [SPECIFIC, GENERIC] - An Event is SPECIFIC if it is understood as a singular occurrence at a particular place and time, or a finite set of such occurrences.
            event['Genericity']
            # Modality [ASSERTED, OTHER] - An Event is ASSERTED when the author or speaker makes reference to it as though it were a real occurrence.
            event['Modality']

            # Types
            # LIFE
            life = event(name='LIFE')
            # LIFE.BE-BORN - A BE-BORN Event occurs whenever a PERSON Entity is given birth to.
            be_born = life(name='BE-BORN')
            be_born.involve(person)
            # LIFE.MARRY - MARRY Events are official Events, where two people are married under the legal definition.
            marry = life(name='MARRY')
            # marry.involve(person)  # not documented explicitly
            # LIFE.DIVORCE - A DIVORCE Event occurs whenever two people are officially divorced under the legal definition of divorce.
            divorce = life(name='DIVORCE')
            # divorce.involve(person)  # no document
            # LIFE.INJURE - An INJURE Event occurs whenever a PERSON Entity experiences physical harm.
            injure = life(name='INJURE')
            injure.involve(person)
            # LIFE.DIE - A DIE Event occurs whenever the life of a PERSON Entity ends.
            die = life(name='DIE')
            die.involve(person)

            # MOVEMENT
            movement = event(name='MOVEMENT')
            # MOVEMENT.TRANSPORT - A TRANSPORT Event occurs whenever an ARTIFACT (WEAPON or VEHICLE) or a PERSON is moved from one PLACE (GPE, FACILITY, LOCATION) to another.
            transport = movement(name='TRANSPORT')
            # NOTE: not sure how to use `involve`...
            transport.involve(weapon, vehicle, person, gpe, facility, location)

            # TRANSACTION
            transaction = event(name='TRANSACTION')
            # TRANSACTION.TRANSFER-OWNERSHIP - TRANSFER-OWNERSHIP Events refer to the buying, selling, loaning, borrowing, giving, or receiving of artifacts or organizations.
            transfer_ownership = transaction(name='TRANSFER-OWNERSHIP')
            # These Events are taggable only when the thing transferred is known to be a taggable VEHICLE, FACILITY, ORGANIZATION or WEAPON.
            transport.involve(vehicle, facility, organization, weapon)
            # TRANSACTION.TRANSFER-MONEY - TRANSFER-MONEY Events refer to the giving, receiving, borrowing, or lending money when it is not in the context of purchasing something.
            transfer_money = transaction(name='TRANSFER-MONEY')            
            
            # BUSINESS
            business = event(name='BUSINESS')
            # BUSINESS.START-ORG - A START-ORG Event occurs whenever a new ORGANIZATION is created.
            start_org = business(name='START-ORG')
            start_org.involve(organization)
            # BUSINESS.MERGE-ORG - A MERGE-ORG Event occurs whenever two or more ORGANIZATION Entities come together to form a new ORGANIZATION Entity.
            merge_org = business(name='MERGE-ORG')
            merge_org.involve(organization)
            # BUSINESS.DECLARE-BANKRUPTCY - A DECLARE-BANKRUPTCY Event will occur whenever an Entity officially requests legal protection from debt collection due to an extremely negative balance sheet.
            declare_bankruptcy = business(name='DECLARE-BANKRUPTCY')
            # BUSINESS.END-ORG - An END-ORG Event occurs whenever an ORGANIZATION ceases to exist (in other words ‘goes out of business’).
            end_org = business(name='END-ORG')
            end_org.involve(organization)

            # CONFLICT
            conflict = event(name='CONFLICT')
            # CONFLICT.ATTACK - An ATTACK Event is defined as a violent physical act causing harm or damage.
            attack = conflict(name='ATTACK')
            # CONFLICT.DEMONSRATE - A DEMONSRATE Event occurs whenever a large number of people come together in a public area to protest or demand some sort of official action.
            demonstrate = conflict(name='DEMONSRATE')

            # CONTACT
            contact = event(name='CONTACT')
            # CONTACT.MEET - A MEET Event occurs whenever two or more Entities come together at a single location and interact with one another face-to-face.
            meet = contact(name='MEET')
            # CONTACT.PHONE-WRITE - A PHONE-WRITE Event occurs when two or more people directly engage in discussion which does not take place ‘face-to-face’.
            phone_write = contact(name='PHONE-WRITE')
            
            # PERSONELL - All PERSONNEL Events can have an POSITION attribute. The object populating the POSITION-ARG slot in a PERSONNEL Event will be a VALUE of type JOB- TITLE, which consists of a string taken from within the scope of the Event.
            personell = event(name='PERSONELL')
            # NOTE: We do not have VALUE or handle attribute now
            # PERSONELL.START-POSITION - A START-POSITION Event occurs whenever a PERSON Entity begins working for (or changes offices within) an ORGANIZATION or GPE.
            start_position = personell(name='START-POSITION')
            start_position.involve(person, organization, gpe)
            # PERSONELL.END-POSITION - A START-POSITION Event occurs whenever a PERSON Entity begins working for (or changes offices within) an ORGANIZATION or GPE.
            end_position = personell(name='END-POSITION')
            end_position.involve(person, organization, gpe)
            # PERSONELL.NOMINATE - A NOMINATE Event occurs whenever a PERSON is proposed for a START- POSITION Event by the appropriate PERSON, through official channels.
            nominate = personell(name='NOMINATE')
            nominate.involve(person)
            # PERSONELL.ELECT - An ELECT Event occurs whenever a candidate wins an election designed to determine the PERSON argument of a START-POSITION Event.
            elect = personell(name='ELECT')
            elect.involve(person)

            # JUSTICE - Many JUSTICE Events can have a CRIME-ARG attribute. As with the POSITION-ARG in PERSONNEL Events, these argument slots will be filled by Values.
            justice = event(name='JUSTICE')
            # NOTE: We do not have VALUE or handle attribute now
            # JUSTICE.ARREST-JAIL - A JAIL Event occurs whenever the movement of a PERSON is constrained by a state actor (a GPE, its ORGANIZATION subparts, or its PERSON representatives).
            arrest_jail = justice(name='ARREST-JAIL')
            arrest_jail.involve(person, gpe, organization)
            # JUSTICE.RELEASE-PAROLE - A RELEASE Event occurs whenever a state actor (GPE, ORGANIZATION subpart, or PERSON representative) ends its custody of a PERSON Entity.
            release_parole = justice(name='RELEASE-PAROLE')
            release_parole.involve(gpe, organization, person)
            # JUSTICE.TRIAL-HEARING
            # JUSTICE.TRIAL - A TRIAL Event occurs whenever a court proceeding has been initiated for the purposes of determining the guilt or innocence of a PERSON, ORGANIZATION or GPE accused of committing a crime.
            trial = justice(name='TRIAL')
            trial.involve(person, organization, gpe)
            # JUSTICE.HEARING - A HEARING Event occurs whenever a state actor (GPE, ORGANIZATION subpart, or PERSON representative) officially gathers to discuss some criminal legal matter.
            hearing = justice(name='HEARING')
            hearing.involve(person, organization, gpe)
            # A TRIAL-HEARING Event can have a CRIME attribute filled by a string from the text. It is important that the PROSECUTER-ARG be a state actor (GPE, ORGANIZATION subpart or PERSON representing them).
            # NOTE: We do not have VALUE or handle attribute now
            # JUSTICE.CHARGE-INDICT
            # JUSTICE.CHARGE - A CHARGE Event occurs whenever a PERSON, ORGANIZATION or GPE is accused of a crime by a state actor (GPE, an ORGANIZATION subpart of a GPE or a PERSON representing a GPE).
            charge = justice(name='CHARGE')
            charge.involve(person, organization, gpe)
            # JUSTICE.INDICT - An INDICT Event occurs whenever a state actor (GPE, ORG subpart of a GPE or PERSON agent of a GPE) takes official legal action to follow up on an accusation.
            indict = justice(name='INDICT')
            indict.involve(person, organization, gpe)
            # A CHARGE-INDICT Event can have a CRIME-ARG attribute filled by a string from the text.
            # NOTE: We do not have VALUE or handle attribute now
            # JUSTICE.SUE - A SUE Event occurs whenever a court proceeding has been initiated for the purposes of determining the liability of a PERSON, ORGANIZATION or GPE accused of committing a crime or neglecting a commitment.
            sue = justice(name='SUE')
            sue.involve(person, organization, gpe)
            # It can have a CRIME attribute filled by a string from the text. It is not important that the PLAINTIFF-ARG be a state actor (a GPE, an ORGANIZATION subpart or a PERSON representing them).
            # NOTE: We do not have VALUE or handle attribute now
            # JUSTICE.CONVICT - A CONVICT Event occurs whenever a TRY Event ends with a successful prosecution of the DEFENDANT-ARG.
            # NOTE: TRY -> TRIAL? 
            convict = justice(name='CONVICT')
            convict.involve(trial)
            # JUSTICE.SENTENCE - A SENTENCE Event takes place whenever the punishment (particularly incarceration) for the DEFENDANT-ARG of a TRY Event is issued by a state actor (a GPE, an ORGANIZATION subpart or a PERSON representing them).
            sentence = justice(name='SENTENCE')
            sentence.involve(trial, gpe, organization, person)
            # It can have a CRIME-ARG attribute filled by a CRIME Value and a SENTENCE-ARG attribute filled by a SENTENCE Value.
            # NOTE: We do not have VALUE or handle attribute now
            # JUSTICE.FINE - A FINE Event takes place whenever a state actor issues a financial punishment to a GPE, PERSON or ORGANIZATION Entity, typically as a result of court proceedings.
            # NOTE: a state actor -> (gpe, organization, person)?
            fine = justice(name='FINE')
            fine.involve(gpe, organization, person)
            # It can have a CRIME attribute filled by a string from the text.
            # NOTE: We do not have VALUE or handle attribute now
            # JUSTICE.EXECUTE - An EXECUTE Event occurs whenever the life of a PERSON is taken by a state actor (a GPE, its ORGANIZATION subparts, or PERSON representatives).
            execute = justice(name='EXECUTE')
            execute.involve(person, gpe, organization, person)
            # It can have a CRIME attribute filled by a string from the text.
            # JUSTICE.EXTRADITE - An EXTRADITE Event occurs whenever a PERSON is sent by a state actor from one PLACE (normally the GPE associated with the state actor, but sometimes a FACILITY under its control) to another place (LOCATION, GPE or FACILITY) for the purposes of legal proceedings there.
            extradite = justice(name='EXTRADITE')
            # PLACE -> GPE, FACILITY
            extradite.involve(person, gpe, facility, location)
            # JUSTICE.ACQUIT - An ACQUIT Event occurs whenever a trial ends but fails to produce a conviction.
            acquit = justice(name='ACQUIT')
            # a trial -> TRAIL?
            acquit.involve(trail)
            # This will include cases where the charges are dropped by the PROSECUTOR-ARG.
            # JUSTICE.APPEAL - An APPEAL Event occurs whenever the decision of a court is taken to a higher court for review.
            appeal = justice(name='APPEAL')
            # JUSTICE.PARDON - A PARDON Event occurs whenever a head-of-state or their appointed representative lifts a sentence imposed by the judiciary.
            pardon = justice(name='PARDON')
