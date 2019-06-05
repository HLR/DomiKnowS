from regr.graph.base import NamedTree, NamedTreeNode


class TestNamedTree(object):
    def test_attach(self):
        # attach(self, sub, name=None):
        nt = NamedTree('A')
        ntn_a = NamedTreeNode('a')
        nt.attach(ntn_a)

        assert ntn_a.name in nt
        assert ntn_a.sup is nt
        assert nt[ntn_a.name] == ntn_a

        ntn_b = NamedTreeNode('b')
        nt.attach(ntn_b, 'ntn_b')

        assert ntn_b.name not in nt
        assert 'ntn_b' in nt
        assert ntn_b.sup is nt
        assert nt['ntn_b'] == ntn_b

        nt_b = NamedTree('B')
        nt.attach(nt_b)

        assert nt_b.name in nt
        assert nt_b.sup is nt
        assert nt[nt_b.name] == nt_b

        assert len(nt) == 3

        keys = list(nt.keys())

        assert keys[0] == ntn_a.name
        assert keys[1] == 'ntn_b'
        assert keys[2] == nt_b.name

    def test_detach_sub(self):
        # detach(self, sub=None, all=False)
        nt = NamedTree('A')
        ntn_a = NamedTreeNode('a')
        nt.attach(ntn_a)
        ntn_b = NamedTreeNode('b')
        nt.attach(ntn_b)
        nt_b = NamedTree('B')
        nt.attach(nt_b)
        nt.detach(ntn_a)

        assert ntn_a.name not in nt
        assert ntn_b.name in nt
        assert nt_b.name in nt
        assert ntn_a.sup is None
        assert ntn_b.sup is nt
        assert nt_b.sup is nt
        assert len(nt) == 2

    def test_detach_none(self):
        # detach(self, sub=None, all=False)
        nt = NamedTree('A')
        ntn_a = NamedTreeNode('a')
        nt.attach(ntn_a)
        ntn_b = NamedTreeNode('b')
        nt.attach(ntn_b)
        nt_b = NamedTree('B')
        nt.attach(nt_b)
        nt.detach()

        assert ntn_a.name not in nt
        assert ntn_b.name not in nt
        assert nt_b.name in nt
        assert ntn_a.sup is None
        assert ntn_b.sup is None
        assert nt_b.sup is nt
        assert len(nt) == 1

    def test_detach_all(self):
        # detach(self, sub=None, all=False)
        nt = NamedTree('A')
        ntn_a = NamedTreeNode('a')
        nt.attach(ntn_a)
        ntn_b = NamedTreeNode('b')
        nt.attach(ntn_b)
        nt_b = NamedTree('B')
        nt.attach(nt_b)
        nt.detach(all=True)

        assert ntn_a.name not in nt
        assert ntn_b.name not in nt
        assert nt_b.name not in nt
        assert ntn_a.sup is None
        assert ntn_b.sup is None
        assert nt_b.sup is None
        assert len(nt) == 0

    def test_getitem(self):
        nt_a = NamedTree('A')
        ntn_a = NamedTreeNode('a')
        nt_a.attach(ntn_a)
        nt_b = NamedTree('B')
        nt_a.attach(nt_b)
        ntn_b = NamedTreeNode('b')
        nt_b.attach(ntn_b)

        assert nt_a['a'] == ntn_a
        assert nt_a['B'] == nt_b
        assert nt_b['b'] == ntn_b
        assert nt_a['B', 'b'] == ntn_b
        assert nt_a['B/b'] == ntn_b

    def test_setitem(self):
        nt = NamedTree('A')
        ntn_a = NamedTreeNode('a')
        nt_b = NamedTree('B')
        obj_c = object()

        nt['a'] = ntn_a
        nt['b'] = nt_b
        nt['c'] = obj_c

        assert nt['a'] == ntn_a
        assert nt['b'] == nt_b
        assert nt['c'] == obj_c
        assert len(nt) == 3

        keys = list(nt.keys())

        assert keys[0] == 'a'
        assert keys[1] == 'b'
        assert keys[2] == 'c'

    def test_delitem(self):
        nt = NamedTree('A')
        ntn_a = NamedTreeNode('a')
        nt_b = NamedTree('B')
        obj_c = object()

        nt['a'] = ntn_a
        nt['b'] = nt_b
        nt['c'] = obj_c

        del nt['b']

        assert nt['a'] == ntn_a
        assert 'b' not in nt
        assert nt['c'] == obj_c
        assert len(nt) == 2

        keys = list(nt.keys())

        assert keys[0] == 'a'
        assert keys[1] == 'c'

    def test_what(self):
        nt_a = NamedTree('A')
        ntn_a = NamedTreeNode('a')
        nt_a.attach(ntn_a)
        nt_b = NamedTree('B')
        nt_a.attach(nt_b)
        ntn_b = NamedTreeNode('b')
        nt_b.attach(ntn_b)

        assert repr(nt_a) == \
            "NamedTree(name='A', what={'subs': {'B': NamedTree(name='B', what={'subs': {'b': NamedTreeNode(name='b', what={'sup': NamedTree(name='B')})},\n 'sup': NamedTree(name='A')}),\n          'a': NamedTreeNode(name='a', what={'sup': NamedTree(name='A')})},\n 'sup': None})"

        assert repr(nt_b) == \
            "NamedTree(name='B', what={'subs': {'b': NamedTreeNode(name='b', what={'sup': NamedTree(name='B')})},\n 'sup': NamedTree(name='A', what={'subs': {'B': NamedTree(name='B'),\n          'a': NamedTreeNode(name='a', what={'sup': NamedTree(name='A')})},\n 'sup': None})})"
