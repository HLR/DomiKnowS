from regr.graph import Graph, Concept, Relation, Property
# from regr.graph.relation import disjoint
from regr.graph.logicalConstrain import ifL, nandL, orL, andL


Graph.clear()
Concept.clear()
Relation.clear()


with Graph('global') as graph:
    with Graph('linguistic') as ling_graph:
        token = Concept(name='token')
        span_candidate = Concept(name='span_candidate')
        span_annotation = Concept(name='span_annotation')
        anchor_annotation = Concept(name='anchor_annotation')
        token_b = token(name='b')
        token_i = token(name='i')
        token_o = token(name='o')
        nandL(token_b, token_i, token_o)
        span = Concept(name='span')
        document = Concept(name='document')
        span_candidate.has_a(start=token, end=token)
        # span.has_a(start=token, end=token)
        span.equal(span_annotation)
        span.equal(anchor_annotation)
        span.contains(token)
        document.contains(token)
        document.contains(span)
        document.contains(span_annotation)

        pair = Concept(name='pair')
        pair.has_a(span, span)

    with Graph('ACE05') as ace05:
        with Graph('Entities') as entities_graph:
            # ACE (Automatic Content Extraction) English Annotation Guidelines for Entities
            # Version 5.6.6 2006.08.01

            # Entity - Entities may be referenced in a text by their name, indicated by a common noun or noun phrase, or represented by a pronoun.
            entity = span(name='Entity')

            # Entity Class - Each taggable entity must be assigned a class that describes the kind of reference the entity makes to something in the world.
            # TODO: this is a property has a value being one of NEG, SPC, GEN, USP
            # TODO: "must be assigned a class" suggest a constraint that a class is required
            entity['Class'] = Property('Class')
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
            nandL(neg, spc, gen, usp)

            # Types
            # Person - Person entities are limited to humans. A person may be a single individual or a group
            person = entity(name='PER')
            # Organization - Organization entities are limited to corporations, agencies, and other groups of people defined by an established organizational structure.
            organization = entity(name='ORG')
            # GPE (Geo-political Entity) - GPE entities are geographical regions defined by political and/or social groups. A GPE entity subsumes and does not distinguish between a nation, its region, its government, or its people.
            gpe = entity(name='GPE')
            # Location - Location entities are limited to geographical entities such as geographical areas and landmasses, bodies of water, and geological formations.
            location = entity(name='LOC')
            # Facility - Facility entities are limited to buildings and other permanent man-made structures and real estate improvements.
            facility = entity(name='FAC')
            # Vehicle – A vehicle entity is a physical device primarily designed to move an object from one location to another, by (for example) carrying, pulling, or pushing the transported object. Vehicle entities may or may not have their own power source.
            vehicle = entity(name='VEH')
            # Weapon – Weapon entities are limited to physical devices primarily used as instruments for physically harming or destroying other entities.
            weapon = entity(name='WEA')
            # special - timex2
            timex2 = span(name='Timex2')
            # special - value
            value = span(name='value')
            num = value(name='Numeric')
            money = num(name='Money')
            percent = num(name='Percent')
            contact_info = value(name='Contact-Info')
            phone_number = contact_info(name='Phone-Number')
            url = contact_info(name='URL')
            email = contact_info(name='E-Mail')
            job = value(name='Job-Title')
            crime = value(name='Crime')
            sen = value(name='Sentence')

            # disjoint
            nandL(person, organization, gpe, location, facility, vehicle, weapon, timex2, value)

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
            individual = person('PER-Individual')
            # PER.Group - If the Person entity refers to more than one person, tag it as PER.Group unless the group meets the requirements of an Organization or a GPE described below. This will include family names and ethnic and religious groups that do not have a formal organization unifying them.
            group = person('PER-Group')
            # PER.Indefinite - If from the context you can’t judge whether the Person entity refers to one or more than one person, tag it as PER.Indefinite.
            indefinite = person('PER-Indeterminate')
            # disjoint
            nandL(individual, group, indefinite)

            # organization subtypes
            # Government (GOV) - Government organizations are those that are of, relating to, or dealing with the structure or affairs of government, politics, or the state.
            government = organization('ORG-Government')
            # Commercial (COM) - A commercial organization is an entire organization or a taggable portion of an organization that is focused primarily upon providing ideas, products, or services for profit.
            commercial = organization('ORG-Commercial')
            # Educational (EDU) - An educational organization is an entire institution or taggable portion of an institution that is focused primarily upon the furthering or promulgation of learning/education.
            educational = organization('ORG-Educational')
            # Entertainment (ENT) - Entertainment organizations are those whose primary activity is entertainment.
            entertainment = organization('ORG-Entertainment')
            # Non-Governmental Organizations (NonGov) - Non-governmental organizations are those organizations that are not a part of a government or commercial organization and whose main role is advocacy, charity or politics (in a broad sense).
            non_governmental = organization('ORG-Non-Governmental')
            # Media (MED) - Media organizations are those whose primary interest is the distribution of news or publications, regardless of whether the organization is privately or publicly owned.
            media = organization('ORG-Media')
            # Religious (REL) - Religious organizations are those that are primarily devoted to issues of religious worship.
            religious = organization('ORG-Religious')
            # Medical-Science (SCI) - Medical-Science organizations are those whose primary activity is the application of medical care or the pursuit of scientific research, regardless of whether that organization is publicly or privately owned.
            medical_science = organization('ORG-Medical-Science')
            # Sports (SPO) - Sports organizations are those that are primarily concerned with participating in or governing organized sporting events, whether professional, amateur, or scholastic.
            sports = organization('ORG-Sports')
            # NOTE: The collection of organization subtypes is hierarchical in nature. ACE05 assigns the most specific type possible. So the annotated results are disjoint.
            # disjoint
            nandL(government, commercial, educational, entertainment, non_governmental, media, religious, medical_science, sports)

            # GPE subtypes
            # Continent - Taggable mentions of the entireties of any of the seven continents.
            continent = GPE('GPE-Continent')
            # Nation - Taggable mentions of the entireties of any nation.
            nation = GPE('GPE-Nation')
            # State-or-Province - Taggable mentions of the entireties of any state, province, or canton of any nation.
            state_or_province = GPE('GPE-State-or-Province')
            # County-or-District - Taggable mentions of the entireties of any county, district, prefecture, or analogous body of any state/province/canton.
            county_or_district = GPE('GPE-County-or-District')
            # Population-Center - Taggable mentions of the entireties of any GPE below the level of County-or- District.
            population_center = GPE('GPE-Population-Center')
            # GPE-Cluster - Named groupings of GPEs that can function as political entities.
            gpe_cluster = GPE('GPE-GPE-Cluster')
            # Special - A closed set of GPEs for which the conventional labels do not straightforwardly apply.
            special = GPE('GPE-Special')
            # disjoint
            nandL(continent, nation, state_or_province, county_or_district, population_center, gpe_cluster, special)

            # GPE Roles - Annotators need to decide for each entity mention in the text which role (Person, Organization, Location, GPE) the context of that mention invokes. This judgment typically depends on the relations that the entity enters into.
            # TODO: Similar to Class, this is a field required from one the the following four.
            gpe['Role'] = Property('Role')
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
            nandL(gpe_org, gpe_per, gpe_loc, gpe_gpe)

            # Location subtypes
            # Address - A location denoted as a point such as in a postal system or abstract coordinates. The name of a location in a postal system is also an address.
            address = location('LOC-Address')
            # Boundary - A one-dimensional location such as a border between GPE’s or other locations.
            boundary = location('LOC-Boundary')
            # Celestial - A location which is otherworldly or entire-world-inclusive.
            celestial = location('LOC-Celestial')
            # Water-Body - Bodies of water, natural or artificial (man-made).
            water_body = location('LOC-Water-Body')
            # Land-Region-natural - Geologically or ecosystemically designated, non-artificial locations.
            land_region_natural = location('LOC-Land-Region-Natural')
            # Region-International - Taggable locations that cross national borders.
            region_international = location('LOC-Region-International')
            # Region-General - Taggable locations that do not cross national borders.
            region_general = location('LOC-Region-General')
            # disjoint
            nandL(address, boundary, celestial, water_body, land_region_natural, region_international, region_general)

            # Facilities subtypes
            # Airport - A facility whose primary use is as an airport.
            airport = facility('FAC-Airport')
            # Plant - One or more buildings that are used and/or designed solely for industrial purposes: manufacturing, power generation, etc.
            plant = facility('FAC-Plant')
            # Building-or-Grounds - Man-made/-maintained buildings, outdoor spaces, and other such facilities.
            building_or_grounds = facility('FAC-Building-Grounds')
            # Subarea-Facility - Taggable portions of facilities. The threshold of taggability of subarea-facility is the ability of the area to contain a normally proportioned person comfortably.
            subarea_facility = facility('FAC-Subarea-Facility')
            # Path - A facility that allows fluids, energies, persons or vehicles to pass from one location to another.
            path = facility('FAC-Path')
            # disjoint
            nandL(airport, plant, building_or_grounds, subarea_facility, path)

            # Vehicle subtypes
            # Air - Vehicles designed to locomote primarily through the air, not touching water or land.
            air = vehicle('VEH-Air')
            # Land - Vehicles designed to locomote primarily upon land.
            land = vehicle('VEH-Land')
            # Water - Vehicles designed to locomote primarily on or submerged in water.
            water = vehicle('VEH-Water')
            # Subarea-Vehicle - A portion of a Vehicle entity that is of a size such that humans can fit inside with some degree of comfort.
            subarea_vehicle = vehicle('VEH-Subarea-Vehicle')
            # Underspecified - Vehicles whose subtype is not specified in the text, or sets of vehicles of different subtypes.
            underspecified_vihicle = vehicle('VEH-Underspecified')

            # Weapon subtypes
            # Blunt - Blunt weapons are those designed or used as bludgeoning instruments.
            blunt = weapon('WEA-Blunt')
            # Exploding - Exploding weapons are those that are designed or used to accomplish damage through explosion.
            exploding = weapon('WEA-Exploding')
            # Sharp - Sharp weapons are those designed or used to cut, slash, jab, & hack.
            sharp = weapon('WEA-Sharp')
            # Chemical - A chemical weapon is any device or substance that is designed or has been used for the purpose of causing death or serious injury through the release, dissemination or impact of toxic or poisonous chemicals or their immediate precursors.
            chemical = weapon('WEA-Chemical')
            # Biological - Biological weapons are bacteria, viruses, fungi, toxins, as well as the means of their dispersal, used for the spread of disease among humans, plants & animals.
            biological = weapon('WEA-Biological')
            # Shooting - Shooting weapons are weapons designed or used to send projectile objects at great speed for the purpose of causing harm.
            shooting = weapon('WEA-Shooting')
            # Projectile - Projectile weapons are weapons designed or used to be projected at great speed for the purpose of causing harm.
            projectile = weapon('WEA-Projectile')
            # Nuclear - Nuclear weapons are those designed or used for the purpose of causing damage, death, and harm through the expenditure of radiological or nuclear energies.
            nuclear = weapon('WEA-Nuclear')
            # Underspecified - Underspecified weapons are weapons whose subtype is not specified in the text, or sets of weapons of different subtypes.
            underspecified_weapon = weapon('WEA-Underspecified')

        with Graph('Relations') as relations_graph:
            # ACE (Automatic Content Extraction) English Annotation Guidelines for Relations
            # Version 5.8.3 – 2005.07.01

            # Relation - Relations are ordered pairs of entities.
            relation = pair(name='Relation')
            relation.has_a(arg1=entity, arg2=entity)

            # Modality
            relation['Modality'] = Property('Modality')
            # Asserted - when the Reasonable Reader Rule is interpreted relative to the 'Real' world;
            # Other - when the Reasonable Reader Rule is taken to hold in a particular counterfactual world.
            
            # Tense - TENSE will be defined relative to the time of speech.
            relation['Tense'] = Property('Tense')
            # Past - the Relation is taken to hold only for some span prior to the time of speech;
            # Future - the Relation is taken to hold only for some span after the time of speech;
            # Present - the Relation is taken to hold for a limited time overlapping with the time of speech;
            # Unspecified - the Relation is ‘static’ or the span of time for which it holds cannot be determined with certainty;

            # Classes
            relation['Class'] = Property('Class')
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
            physical = relation(name='PHYS')
            # Physical.Located - The Located Relation captures the physical location of an entity.
            located = physical(name='Located')
            # located.has_a(arg1=person, arg2=entity)
            # arg2 is one of FAC, LOC, GPE
            ifL(located, ('x', 'y'), andL(PER, ('x',), orL(FAC, LOC, GPE, ('y',))))
            # Physical.Near - Near indicates that an entity is explicitly near another entity, but neither entity is a part of the other or located in/at the other.
            near = physical(name='Near')
            # near.has_a(arg1=entity, arg2=entity)
            # arg1 is one of PER, FAC, GPE, LOC
            # arg2 is one of FAC, GPE, LOC
            ifL(near, ('x', 'y'), andL(orL(PER, FAC, GPE, LOC, ('x',)), orL(FAC, GPE, LOC, ('y',))))

            # Part-whole
            part_whole = relation(name='PART-WHOLE')
            # Part-whole.Geographical - The Geographical Relation captures the location of a Facility, Location, or GPE in or at or as a part of another Facility, Location, or GPE.
            geographical = part_whole(name='Geographical')
            # geographical.has_a(arg1=entity, arg2=entity)
            # arg1 is one of FAC, LOC, GPE
            # arg2 is one of FAC, LOC, GPE
            ifL(geographical, ('x', 'y'), andL(orL(FAC, LOC, GPE, ('x',)), orL(FAC, LOC, GPE, ('y',))))
            # Part-whole.Subsidiary - Subsidiary captures the ownership, administrative, and other hierarchical relationships between organizations and between organizations and GPEs.
            subsidiary = part_whole(name='Subsidiary')
            # subsidiary.has_a(arg1=organization, arg2=entity)
            # arg2 is one of ORG, GPE
            ifL(subsidiary, ('x', 'y'), andL(ORG, ('x',), orL(ORG, GPE, ('y',))))
            # Part-whole.Artifact - Artifact characterizes physical relationships between concrete physical objects and their parts.
            artifact = part_whole(name='Artifact')
            # artifact.has_a(arg1=entity, arg2=entity)
            # (arg1 is VEH and arg2 is VEH) or (arg1 is WEA and arg2 is WEA)
            ifL(artifact, ('x', 'y'), orL(andL(VEH, ('x',), VEH, ('y',)), andL(WEA, ('x',), WEA, ('y',))))

            # Personal-Social - Personal-Social relations describe the relationship between people. Both arguments must be entities of type PER.
            # The arguments of these Relations are not ordered. The Relations are symmetric.
            personal_social = relation(name='PER-SOC')
            # personal_social.has_a(arg1=person, arg2=person)
            ifL(personal_social, ('x', 'y'), andL(PER, ('x',), PER, ('y',)))
            # Personal-Social.Business - The Business Relation captures the connection between two entities in any professional relationship.
            business = personal_social(name='Business')
            ifL(business, ('x', 'y'), andL(PER, ('x',), PER, ('y',)))
            # Personal-Social.Family - The Family Relation captures the connection between one entity and another with which it is in any familial relationship.
            family = personal_social(name='Family')
            ifL(family, ('x', 'y'), andL(PER, ('x',), PER, ('y',)))
            # Personal-Social.Lasting-Personal - The relationship must involve personal contact (or a reasonable assumption thereof); and there must be some indication or expectation that the relationship exists outside of a particular cited interaction.
            lasting_personal = personal_social(name='Lasting-Personal')
            ifL(lasting_personal, ('x', 'y'), andL(PER, ('x',), PER, ('y',)))
            
            # ORG-Affiliation
            org_affiliation = relation(name='ORG-AFF')
            # ORG-Affiliation.Employment - Employment captures the relationship between Persons and their employers.
            employment = org_affiliation(name='Employment')
            # employment.has_a(arg1=person, arg2=entity)
            # arg2 is one of ORG, GPE
            ifL(employment, ('x', 'y'), andL(PER, ('x',), orL(ORG, GPE, ('y',))))
            # ORG-Affiliation.Ownership - Ownership captures the relationship between a Person and an Organization owned by that Person.
            ownership = org_affiliation(name='Ownership')
            # ownership.has_a(arg1=person, arg2=organization)
            ifL(ownership, ('x', 'y'), andL(PER, ('x',), ORG, ('y',)))
            # ORG-Affiliation.Founder - Founder captures the relationship between an agent (Person, Organization, or GPE) and an Organization or GPE established or set up by that agent.
            founder = org_affiliation(name='Founder')
            # founder.has_a(arg1=entity, arg2=entity)
            # arg1 is one of PER, ORG
            # arg2 is one of ORG, GPE
            ifL(founder, ('x', 'y'), andL(orL(PER, ORG, ('x',)), orL(ORG, GPE, ('y',))))
            # ORG-Affiliation.Student-Alum - Student-Alum captures the relationship between a Person and an educational institution the Person attends or attended.
            student_alum = org_affiliation(name='Student-Alum')
            # student_alum.has_a(arg1=person, arg2=educational)
            ifL(student_alum, ('x', 'y'), andL(PER, ('x',), educational, ('y',)))
            # ORG-Affiliation.Sports-Affiliation - Sports-Affiliation captures the relationship between a player, coach, manager, or assistant and his or her affiliation with a sports organization.
            sports_affiliation = org_affiliation(name='Sports-Affiliation')
            # sports_affiliation.has_a(arg1=person, arg2=organization)
            ifL(sports_affiliation, ('x', 'y'), andL(PER, ('x',), ORG, ('y',)))
            # ORG-Affiliation.Investor-Shareholder - Investor-Shareholder captures the relationship between an agent (Person, Organization, or GPE) and an Organization in which the agent has invested or in which the agent owns shares/stock.
            investor_shareholder = org_affiliation(name='Investor-Shareholder')
            # investor_shareholder.has_a(arg1=entity, arg2=entity)
            # arg1 is one of PER, ORG, GPE
            # arg2 is one of ORG, GPE
            ifL(investor_shareholder, ('x', 'y'), andL(orL(PER, ORG, GPE, ('x',)), orL(ORG, GPE, ('y',))))
            # ORG-Affiliation.Membership - Membership captures the relationship between an agent and an organization of which the agent is a member.
            membership = org_affiliation(name='Membership')
            # membership.has_a(arg1=entity, arg2=organization)
            # arg1 is one of PER, ORG, GPE
            ifL(membership, ('x', 'y'), andL(orL(PER, ORG, GPE, ('x',)), ORG, ('y',)))

            # Agent-Artifact
            agent_artifact = relation('ART')
            # Agent-Artifact.User-Owner-Inventor-Manufacturer - This Relation applies when an agent owns an artifact, has possession of an artifact, uses an artifact, or caused an artifact to come into being.
            user_owner_inventor_manufacturer = agent_artifact(name='User-Owner-Inventor-Manufacturer')
            # user_owner_inventor_manufacturer.has_a(arg1=entity, arg2=entity)
            # arg1 is one of PER, ORG, GPE
            # arg2 is one of WEA, VEH, FAC
            ifL(user_owner_inventor_manufacturer, ('x', 'y'), andL(orL(PER, ORG, GPE, ('x',)), orL(WEA, VEH, FAC, ('y',))))

            # Gen-Affiliation
            gen_affiliation = relation('GEN-AFF')
            # Gen-Affiliation.Citizen-Resident-Religion-Ethnicity - Citizen-Resident-Religion-Ethnicity describes the Relation between a PER entity and PER.Group, LOC, GPE, ORG
            citizen_resident_religion_ethnicity = gen_affiliation('Citizen-Resident-Religion-Ethnicity')
            # citizen_resident_religion_ethnicity.has_a(arg1=person, arg2=entity)
            # arg2 is one of PER.Group, LOC, GPE, ORG
            ifL(citizen_resident_religion_ethnicity, ('x', 'y'), andL(PER, ('x',), orL(group, LOC, GPE, ORG, ('y',))))
            # Gen-Affiliation.Org-Location-Origin - Org-Location-Origin captures the relationship between an organization and the LOC or GPE where it is located, based, or does business.
            org_location_origin = gen_affiliation('Org-Location')
            # org_location_origin.has_a(arg1=organization, arg2=entity)
            # arg2 is one of LOC, GPE
            ifL(org_location_origin, ('x', 'y'), andL(ORG, ('x',), orL(LOC, GPE, ('y',))))

            metonymy = relation('METONYMY')

        with Graph('Events') as events_graph:
            # ACE (Automatic Content Extraction) English Annotation Guidelines for Events
            # Version 5.4.3 2005.07.01

            trigger = span(name='trigger')
            # NOTE: do we need the abstract event or base it on trigger?
            # event = Concept(name='Event')
            # event.has_a(trigger)
            # NOTE: Instead of predicting event, we predict trigger
            #       Here event is just a alias of trigger
            event = trigger

            # @Concept.relation_type('involve')
            # class Involve(Relation): pass
            # event.involve(participant)

            # There can be variable number of participant(s) and attribute(s)
            # Create concept based on pair so that we can predit them
            # Here we define the base concepts
            event_argument = pair(name='EventArgument')
            event_argument.has_a(event=event, argument=span)
            participant_argument = event_argument(name='Participant')
            attribute_argument = event_argument(name='Attribute')
            # and a shortcut function to create role concept and rule
            def involve(event_type, argument_type, **kwargs):
                for role, concepts in kwargs.items():
                    role_argument = argument_type(name=f'{event_type.name}-{role}')
                    if isinstance(concepts, Concept):
                        ifL(role_argument, ('x', 'y'), andL(event_type, ('x',), concepts, ('y',)))
                    elif isinstance(concepts, (tuple, list)):
                        ifL(role_argument, ('x', 'y'), andL(event_type, ('x',), orL(*concepts, ('y',))))
                    else:
                        raise TypeError('Argument must be Concept or tuple of Concepts.')

            # Polarity [POSITIVE, NEGATIVE]- An Event is NEGATIVE when it is explicitly indicated that the Event did not occur (see examples). All other Events are POSITIVE.
            event['Polarity'] = Property('Polarity')
            # Tense [PAST, FUTURE, PRESENT] - TENSE is determined with respect to the speaker or author.
            event['Tense'] = Property('Tense')
            # Genericity [SPECIFIC, GENERIC] - An Event is SPECIFIC if it is understood as a singular occurrence at a particular place and time, or a finite set of such occurrences.
            event['Genericity'] = Property('Genericity')
            # Modality [ASSERTED, OTHER] - An Event is ASSERTED when the author or speaker makes reference to it as though it were a real occurrence.
            event['Modality'] = Property('Modality')

            # Types
            # LIFE
            life = event(name='Life')
            # LIFE.BE-BORN - A BE-BORN Event occurs whenever a PERSON Entity is given birth to.
            be_born = life(name='Be-Born')
            # S6 - participant: Person: PER
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # be_born.involve(PER, timex2, GPE, LOC, FAC)
            involve(be_born, participant_argument, Person=PER)
            involve(be_born, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))
            # LIFE.MARRY - MARRY Events are official Events, where two people are married under the legal definition.
            marry = life(name='Marry')
            # marry.involve(person)  # not documented explicitly
            # S6 - participant: Person: PER
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # marry.involve(PER, timex2, GPE, LOC, FAC)
            involve(marry, participant_argument, Person=PER)
            involve(marry, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))
            # LIFE.DIVORCE - A DIVORCE Event occurs whenever two people are officially divorced under the legal definition of divorce.
            divorce = life(name='Divorce')
            # divorce.involve(person)  # no document
            # S6 - participant: Person: PER
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # divorce.involve(PER, timex2, GPE, LOC, FAC)
            involve(divorce, participant_argument, Person=PER)
            involve(divorce, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))
            # LIFE.INJURE - An INJURE Event occurs whenever a PERSON Entity experiences physical harm.
            injure = life(name='Injure')
            # injure.involve(person)
            # S6 - participant: Agent: (PER, ORG, GPE), Victim: PER, Instrument: (WEA, VEH)
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # injure.involve(PER, ORG, GPE, WEA, VEH, timex2, GPE, LOC, FAC)
            involve(injure, participant_argument, Agent=(PER, ORG, GPE), Victim=PER, Instrument=(WEA, VEH))
            involve(injure, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))
            # LIFE.DIE - A DIE Event occurs whenever the life of a PERSON Entity ends.
            die = life(name='Die')
            # die.involve(person)
            # S6 - participant: Agent: (PER, ORG, GPE), Victim: PER, Instrument: (WEA, VEH)
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # die.involve(PER, ORG, GPE, WEA, VEH, timex2, GPE, LOC, FAC)
            involve(die, participant_argument, Agent=(PER, ORG, GPE), Victim=PER, Instrument=(WEA, VEH))
            involve(die, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))

            # MOVEMENT
            movement = event(name='Movement')
            # MOVEMENT.TRANSPORT - A TRANSPORT Event occurs whenever an ARTIFACT (WEAPON or VEHICLE) or a PERSON is moved from one PLACE (GPE, FACILITY, LOCATION) to another.
            transport = movement(name='Transport')
            # transport.involve(weapon, vehicle, person, gpe, facility, location)
            # S6 - participant: Agent: (PER, ORG, GPE), Artifact: (PER, WEA, VEH), Vehicle: VEH, Price: num, Origin: (GPE, LOC, FAC), Destination: (GPE, LOC, FAC)
            # S6 - attribute: Time: Time
            # transport.involve(PER, ORG, GPE, WEA, VEH, num, LOC, FAC, timex2)
            involve(transport, participant_argument, Agent=(PER, ORG, GPE), Artifact=(PER, WEA, VEH), Vehicle=VEH, Price=num, Origin=(GPE, LOC, FAC), Destination=(GPE, LOC, FAC))
            involve(transport, attribute_argument, Time=timex2)

            # TRANSACTION
            transaction = event(name='Transaction')
            # TRANSACTION.TRANSFER-OWNERSHIP - TRANSFER-OWNERSHIP Events refer to the buying, selling, loaning, borrowing, giving, or receiving of artifacts or organizations.
            transfer_ownership = transaction(name='Transfer-Ownership')
            # These Events are taggable only when the thing transferred is known to be a taggable VEHICLE, FACILITY, ORGANIZATION or WEAPON.
            # transport.involve(vehicle, facility, organization, weapon)
            # S6 - participant: Buyer: (PER, ORG, GPE), Seller: (PER, ORG, GPE), Beneficiary: (PER, ORG, GPE), Artifact: (VEH, WEA, FAC, ORG), Price: money
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # transfer_ownership.involve(PER, ORG, GPE, VEH, WEA, money, timex2, LOC, FAC)
            involve(transfer_ownership, participant_argument, Buyer=(PER, ORG, GPE), Seller=(PER, ORG, GPE), Beneficiary=(PER, ORG, GPE), Artifact=(VEH, WEA, FAC, ORG), Price=money)
            involve(transfer_ownership, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))
            # TRANSACTION.TRANSFER-MONEY - TRANSFER-MONEY Events refer to the giving, receiving, borrowing, or lending money when it is not in the context of purchasing something.
            transfer_money = transaction(name='Transfer-Money')
            # S6 - participant: Giver: (PER, ORG, GPE), Recipient: (PER, ORG, GPE), Beneficiary: (PER, ORG, GPE), Money: money
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # transfer_money.involve(PER, ORG, GPE, money, timex2, LOC, FAC)
            involve(transfer_money, participant_argument, Giver=(PER, ORG, GPE), Recipient=(PER, ORG, GPE), Beneficiary=(PER, ORG, GPE), Money=money)
            involve(transfer_money, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))

            # BUSINESS
            business = event(name='Business-Event')
            # BUSINESS.START-ORG - A START-ORG Event occurs whenever a new ORGANIZATION is created.
            start_org = business(name='Start-Org')
            # start_org.involve(organization)
            # S6 - participant: Agent: (PER, ORG, GPE), Org: ORG
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # start_org.involve(PER, ORG, GPE, timex2, LOC, FAC)
            involve(start_org, participant_argument, Agent=(PER, ORG, GPE), Org=ORG)
            involve(start_org, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))
            # BUSINESS.MERGE-ORG - A MERGE-ORG Event occurs whenever two or more ORGANIZATION Entities come together to form a new ORGANIZATION Entity.
            merge_org = business(name='Merge-Org')
            # merge_org.involve(organization)
            # S6 - participant: Org: ORG
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # merge_org.involve(ORG, timex2, GPE, LOC, FAC)
            involve(merge_org, participant_argument, Org=ORG)
            involve(merge_org, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))
            # BUSINESS.DECLARE-BANKRUPTCY - A DECLARE-BANKRUPTCY Event will occur whenever an Entity officially requests legal protection from debt collection due to an extremely negative balance sheet.
            declare_bankruptcy = business(name='Declare-Bankruptcy')
            # S6 - participant: Org: (ORG, PER, GPE)
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # declare_bankruptcy.involve(ORG, PER, GPE, timex2, LOC, FAC)
            involve(declare_bankruptcy, participant_argument, Org=(ORG, PER, GPE))
            involve(declare_bankruptcy, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))
            # BUSINESS.END-ORG - An END-ORG Event occurs whenever an ORGANIZATION ceases to exist (in other words ‘goes out of business’).
            end_org = business(name='End-Org')
            # end_org.involve(organization)
            # S6 - participant: Org: ORG
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # end_org.involve(ORG, timex2, GPE, LOC, FAC)
            involve(end_org, participant_argument, Org=ORG)
            involve(end_org, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))

            # CONFLICT
            conflict = event(name='Conflict')
            # CONFLICT.ATTACK - An ATTACK Event is defined as a violent physical act causing harm or damage.
            attack = conflict(name='Attack')
            # S6 - participant: Attacker: (PER, ORG, GPE), Target: (PER, ORG, VEH, FAC, WEA), Instrument: (WEA, VEH)
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # attack.involve(PER, ORG, GPE, VEH, FAC, WEA, timex2, LOC, FAC)
            involve(attack, participant_argument, Attacker=(PER, ORG, GPE), Target=(PER, ORG, VEH, FAC, WEA), Instrument=(WEA, VEH))
            involve(attack, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))
            # CONFLICT.DEMONSRATE - A DEMONSRATE Event occurs whenever a large number of people come together in a public area to protest or demand some sort of official action.
            demonstrate = conflict(name='Demonstrate')
            # S6 - participant: Entity: (PER, ORG)
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # demonstrate.involve(PER, ORG, timex2, GPE, LOC, FAC)
            involve(demonstrate, participant_argument, Entity=(PER, ORG))
            involve(demonstrate, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))

            # CONTACT
            contact = event(name='Contact')
            # CONTACT.MEET - A MEET Event occurs whenever two or more Entities come together at a single location and interact with one another face-to-face.
            meet = contact(name='Meet')
            # S6 - participant: Entity: (PER, ORG, GPE)
            # S6 - attribute: Time: Time, Place: (GPE, LOC, FAC)
            # meet.involve(PER, ORG, GPE, timex2, LOC, FAC)
            involve(meet, participant_argument, Entity=(PER, ORG, GPE))
            involve(meet, attribute_argument, Time=timex2, Place=(GPE, LOC, FAC))
            # CONTACT.PHONE-WRITE - A PHONE-WRITE Event occurs when two or more people directly engage in discussion which does not take place ‘face-to-face’.
            phone_write = contact(name='Phone-Write')
            # S6 - participant: Entity: (PER, ORG, GPE)
            # S6 - attribute: Time: Time
            # phone_write.involve(PER, ORG, GPE, timex2)
            involve(phone_write, participant_argument, Entity=(PER, ORG, GPE))
            involve(phone_write, attribute_argument, Time=timex2)
            
            # PERSONELL - All PERSONNEL Events can have an POSITION attribute. The object populating the POSITION-ARG slot in a PERSONNEL Event will be a VALUE of type JOB- TITLE, which consists of a string taken from within the scope of the Event.
            personell = event(name='Personnel')
            # NOTE: We do not have VALUE or handle attribute now
            # PERSONELL.START-POSITION - A START-POSITION Event occurs whenever a PERSON Entity begins working for (or changes offices within) an ORGANIZATION or GPE.
            start_position = personell(name='Start-Position')
            # start_position.involve(person, organization, gpe)
            # S6 - participant: Person: PER, Entity: (ORG, GPE)
            # S6 - attribute: Position: JOB, Time: Time, Place: (GPE, LOC, FAC)
            # start_position.involve(PER, ORG, GPE, job, timex2, LOC, FAC)
            involve(start_position, participant_argument, Person=PER, Entity=(ORG, GPE))
            involve(start_position, attribute_argument, Position=job, Time=timex2, Place=(GPE, LOC, FAC))
            # PERSONELL.END-POSITION - A START-POSITION Event occurs whenever a PERSON Entity begins working for (or changes offices within) an ORGANIZATION or GPE.
            end_position = personell(name='End-Position')
            # end_position.involve(person, organization, gpe)
            # S6 - participant: Person: PER, Entity: (ORG, GPE)
            # S6 - attribute: Position: JOB, Time: Time, Place: (GPE, LOC, FAC)
            # end_position.involve(PER, ORG, GPE, job, timex2, LOC, FAC)
            involve(end_position, participant_argument, Person=PER, Entity=(ORG, GPE))
            involve(end_position, attribute_argument, Position=job, Time=timex2, Place=(GPE, LOC, FAC))
            # PERSONELL.NOMINATE - A NOMINATE Event occurs whenever a PERSON is proposed for a START- POSITION Event by the appropriate PERSON, through official channels.
            nominate = personell(name='Nominate')
            # nominate.involve(person)
            # S6 - participant: Person: PER, Agent: (PER, ORG, GPE, FAC)
            # S6 - attribute: Position: JOB, Time: Time, Place: (GPE, LOC, FAC)
            # nominate.involve(PER, ORG, GPE, FAC, job, timex2, LOC)
            involve(nominate, participant_argument, Person=PER, Agent=(PER, ORG, GPE, FAC))
            involve(nominate, attribute_argument, Position=job, Time=timex2, Place=(GPE, LOC, FAC))
            # PERSONELL.ELECT - An ELECT Event occurs whenever a candidate wins an election designed to determine the PERSON argument of a START-POSITION Event.
            elect = personell(name='Elect')
            # elect.involve(person)
            # S6 - participant: Person: PER, Entity: (PER, ORG, GPE)
            # S6 - attribute: Position: JOB, Time: Time, Place: (GPE, LOC, FAC)
            # elect.involve(PER, ORG, GPE, job, timex2, LOC, FAC)
            involve(elect, participant_argument, Person=PER, Entity=(PER, ORG, GPE))
            involve(elect, attribute_argument, Position=job, Time=timex2, Place=(GPE, LOC, FAC))

            # JUSTICE - Many JUSTICE Events can have a CRIME-ARG attribute. As with the POSITION-ARG in PERSONNEL Events, these argument slots will be filled by Values.
            justice = event(name='Justice')
            # JUSTICE.ARREST-JAIL - A JAIL Event occurs whenever the movement of a PERSON is constrained by a state actor (a GPE, its ORGANIZATION subparts, or its PERSON representatives).
            arrest_jail = justice(name='Arrest-Jail')
            # arrest_jail.involve(person, gpe, organization)
            # S6 - participant: Person: PER, Agent: (PER, ORG, GPE)
            # S6 - attribute: Crime: CRIME, Time: Time, Place: (GPE, LOC, FAC)
            # arrest_jail.involve(PER, ORG, GPE, crime, timex2, LOC, FAC)
            involve(arrest_jail, participant_argument, Person=PER, Agent=(PER, ORG, GPE))
            involve(arrest_jail, attribute_argument, Crime=crime, Time=timex2, Place=(GPE, LOC, FAC))
            # JUSTICE.RELEASE-PAROLE - A RELEASE Event occurs whenever a state actor (GPE, ORGANIZATION subpart, or PERSON representative) ends its custody of a PERSON Entity.
            release_parole = justice(name='Release-Parole')
            # release_parole.involve(gpe, organization, person)
            # S6 - participant: Person: PER, Entity: (PER, ORG, GPE)
            # S6 - attribute: Crime: CRIME, Time: Time, Place: (GPE, LOC, FAC)
            # release_parole.involve(PER, ORG, GPE, crime, timex2, LOC, FAC)
            involve(release_parole, participant_argument, Person=PER, Entity=(PER, ORG, GPE))
            involve(release_parole, attribute_argument, Crime=crime, Time=timex2, Place=(GPE, LOC, FAC))
            # JUSTICE.TRIAL-HEARING
            # JUSTICE.TRIAL - A TRIAL Event occurs whenever a court proceeding has been initiated for the purposes of determining the guilt or innocence of a PERSON, ORGANIZATION or GPE accused of committing a crime.
            # JUSTICE.HEARING - A HEARING Event occurs whenever a state actor (GPE, ORGANIZATION subpart, or PERSON representative) officially gathers to discuss some criminal legal matter.
            trial = justice(name='Trial-Hearing')
            # trial.involve(person, organization, gpe)
            # S6 - participant: Defendant: (PER, ORG, GPE), Prosecutor: (PER, ORG, GPE), Adjudicator: (PER, ORG, GPE)
            # S6 - attribute: Crime: CRIME, Time: Time, Place: (GPE, LOC, FAC)
            # trial.involve(PER, ORG, GPE, crime, timex2, LOC, FAC)
            involve(trial, participant_argument, Defendant=(PER, ORG, GPE), Prosecutor=(PER, ORG, GPE), Adjudicator=(PER, ORG, GPE))
            involve(trial, attribute_argument, Crime=crime, Time=timex2, Place=(GPE, LOC, FAC))
            # JUSTICE.CHARGE-INDICT
            # JUSTICE.CHARGE - A CHARGE Event occurs whenever a PERSON, ORGANIZATION or GPE is accused of a crime by a state actor (GPE, an ORGANIZATION subpart of a GPE or a PERSON representing a GPE).
            # JUSTICE.INDICT - An INDICT Event occurs whenever a state actor (GPE, ORG subpart of a GPE or PERSON agent of a GPE) takes official legal action to follow up on an accusation.
            charge = justice(name='Charge-Indict')
            # charge.involve(person, organization, gpe)
            # S6 - participant: Defendant: (PER, ORG, GPE), Prosecutor: (PER, ORG, GPE), Adjudicator: (PER, ORG, GPE)
            # S6 - attribute: Crime: CRIME, Time: Time, Place: (GPE, LOC, FAC)
            # charge.involve(PER, ORG, GPE, crime, timex2, LOC, FAC)
            involve(charge, participant_argument, Defendant=(PER, ORG, GPE), Prosecutor=(PER, ORG, GPE), Adjudicator=(PER, ORG, GPE))
            involve(charge, attribute_argument, Crime=crime, Time=timex2, Place=(GPE, LOC, FAC))
            # JUSTICE.SUE - A SUE Event occurs whenever a court proceeding has been initiated for the purposes of determining the liability of a PERSON, ORGANIZATION or GPE accused of committing a crime or neglecting a commitment.
            sue = justice(name='Sue')
            # sue.involve(person, organization, gpe)
            # S6 - participant: Plaintiff: (PER, ORG, GPE), Defendant: (PER, ORG, GPE), Adjudicator: (PER, ORG, GPE)
            # S6 - attribute: Crime: CRIME, Time: Time, Place: (GPE, LOC, FAC)
            # sue.involve(PER, ORG, GPE, crime, timex2, LOC, FAC)
            involve(sue, participant_argument, Plaintiff=(PER, ORG, GPE), Defendant=(PER, ORG, GPE), Adjudicator=(PER, ORG, GPE))
            involve(sue, attribute_argument, Crime=crime, Time=timex2, Place=(GPE, LOC, FAC))
            # It can have a CRIME attribute filled by a string from the text. It is not important that the PLAINTIFF-ARG be a state actor (a GPE, an ORGANIZATION subpart or a PERSON representing them).
            # JUSTICE.CONVICT - A CONVICT Event occurs whenever a TRY Event ends with a successful prosecution of the DEFENDANT-ARG.
            # NOTE: TRY -> TRIAL? 
            convict = justice(name='Convict')
            # convict.involve(trial)
            # S6 - participant: Defendant: (PER, ORG, GPE), Adjudicator: (PER, ORG, GPE)
            # S6 - attribute: Crime: CRIME, Time: Time, Place: (GPE, LOC, FAC)
            # convict.involve(PER, ORG, GPE, crime, timex2, LOC, FAC)
            involve(convict, participant_argument, Defendant=(PER, ORG, GPE), Adjudicator=(PER, ORG, GPE))
            involve(convict, attribute_argument, Crime=crime, Time=timex2, Place=(GPE, LOC, FAC))
            # JUSTICE.SENTENCE - A SENTENCE Event takes place whenever the punishment (particularly incarceration) for the DEFENDANT-ARG of a TRY Event is issued by a state actor (a GPE, an ORGANIZATION subpart or a PERSON representing them).
            sentence = justice(name='Sentence-Event')
            # sentence.involve(trial, gpe, organization, person)
            # S6 - participant: Defendant: (PER, ORG, GPE), Adjudicator: (PER, ORG, GPE)
            # S6 - attribute: Crime: CRIME, Sentence: SEN, Time: Time, Place: (GPE, LOC, FAC)
            # sentence.involve(PER, ORG, GPE, crime, sen, timex2, LOC, FAC)
            involve(sentence, participant_argument, Defendant=(PER, ORG, GPE), Adjudicator=(PER, ORG, GPE))
            involve(sentence, attribute_argument, Crime=crime, Sentence=sen, Time=timex2, Place=(GPE, LOC, FAC))
            # It can have a CRIME-ARG attribute filled by a CRIME Value and a SENTENCE-ARG attribute filled by a SENTENCE Value.
            # JUSTICE.FINE - A FINE Event takes place whenever a state actor issues a financial punishment to a GPE, PERSON or ORGANIZATION Entity, typically as a result of court proceedings.
            # NOTE: a state actor -> (gpe, organization, person)?
            fine = justice(name='Fine')
            # fine.involve(gpe, organization, person)
            # S6 - participant: Entity: (PER, ORG, GPE), Adjudicator: (PER, ORG, GPE), Money: NUM
            # S6 - attribute: Crime: CRIME, Time: Time, Place: (GPE, LOC, FAC)
            # fine.involve(PER, ORG, GPE, num, crime, timex2, LOC, FAC)
            involve(fine, participant_argument, Entity=(PER, ORG, GPE), Adjudicator=(PER, ORG, GPE), Money=num)
            involve(fine, attribute_argument, Crime=crime, Time=timex2, Place=(GPE, LOC, FAC))
            # It can have a CRIME attribute filled by a string from the text.
            # JUSTICE.EXECUTE - An EXECUTE Event occurs whenever the life of a PERSON is taken by a state actor (a GPE, its ORGANIZATION subparts, or PERSON representatives).
            execute = justice(name='Execute')
            # execute.involve(person, gpe, organization, person)
            # S6 - participant: Person: PER, Agent: (PER, ORG, GPE)
            # S6 - attribute: Crime: CRIME, Time: Time, Place: (GPE, LOC, FAC)
            # execute.involve(PER, ORG, GPE, crime, timex2, LOC, FAC)
            involve(execute, participant_argument, Person=PER, Agent=(PER, ORG, GPE))
            involve(execute, attribute_argument, Crime=crime, Time=timex2, Place=(GPE, LOC, FAC))
            # It can have a CRIME attribute filled by a string from the text.
            # JUSTICE.EXTRADITE - An EXTRADITE Event occurs whenever a PERSON is sent by a state actor from one PLACE (normally the GPE associated with the state actor, but sometimes a FACILITY under its control) to another place (LOCATION, GPE or FACILITY) for the purposes of legal proceedings there.
            extradite = justice(name='Extradite')
            # PLACE -> GPE, FACILITY
            # extradite.involve(person, gpe, facility, location)
            # S6 - participant: Agent: (PER, ORG, GPE), Person: PER, Destination: (GPE, LOC, FAC), Origin: (GPE, LOC, FAC)
            # S6 - attribute: Crime: CRIME, Time: Time
            # extradite.involve(PER, ORG, GPE, LOC, FAC, crime, timex2)
            involve(extradite, participant_argument, Agent=(PER, ORG, GPE), Person=PER, Destination=(GPE, LOC, FAC), Origin=(GPE, LOC, FAC))
            involve(extradite, attribute_argument, Crime=crime, Time=timex2)
            # JUSTICE.ACQUIT - An ACQUIT Event occurs whenever a trial ends but fails to produce a conviction.
            acquit = justice(name='Acquit')
            # a trial -> TRAIL?
            # acquit.involve(trail)
            # S6 - participant: Defendant: (PER, ORG, GPE), Adjudicator: (PER, ORG, GPE)
            # S6 - attribute: Crime: CRIME, Time: Time, Place: (GPE, LOC, FAC)
            # acquit.involve(PER, ORG, GPE, crime, timex2, LOC, FAC)
            involve(acquit, participant_argument, Defendant=(PER, ORG, GPE), Adjudicator=(PER, ORG, GPE))
            involve(acquit, attribute_argument, Crime=crime, Time=timex2, Place=(GPE, LOC, FAC))
            # This will include cases where the charges are dropped by the PROSECUTOR-ARG.
            # JUSTICE.APPEAL - An APPEAL Event occurs whenever the decision of a court is taken to a higher court for review.
            appeal = justice(name='Appeal')
            # S6 - participant: Defendant: (PER, ORG, GPE), Prosecutor: (PER, ORG, GPE), Adjudicator: (PER, ORG, GPE)
            # S6 - attribute: Crime: CRIME, Time: Time, Place: (GPE, LOC, FAC)
            # appeal.involve(PER, ORG, GPE, crime, timex2, LOC, FAC)
            involve(appeal, participant_argument, Defendant=(PER, ORG, GPE), Prosecutor=(PER, ORG, GPE), Adjudicator=(PER, ORG, GPE))
            involve(appeal, attribute_argument, Crime=crime, Time=timex2, Place=(GPE, LOC, FAC))
            # JUSTICE.PARDON - A PARDON Event occurs whenever a head-of-state or their appointed representative lifts a sentence imposed by the judiciary.
            pardon = justice(name='Pardon')
            # S6 - participant: Defendant: (PER, ORG, GPE), Adjudicator: (PER, ORG, GPE)
            # S6 - attribute: Crime: CRIME, Time: Time, Place: (GPE, LOC, FAC)
            # pardon.involve(PER, ORG, GPE, crime, timex2, LOC, FAC)
            involve(pardon, participant_argument, Defendant=(PER, ORG, GPE), Adjudicator=(PER, ORG, GPE))
            involve(pardon, attribute_argument, Crime=crime, Time=timex2, Place=(GPE, LOC, FAC))
