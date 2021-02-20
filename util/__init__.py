from rdflib import Graph, URIRef, BNode, Literal
from torch.nn import Embedding


class StringTripletsBuilder:
    def __init__(self, rdf_graph: Graph):
        self.rdf_graph = rdf_graph
        self._ns_to_id = {}
        self._id_to_ns = {}
        self._ns_id_cntr = 1

    def _next_ns_id(self):
        ns_id = f'ns{self._ns_id_cntr}'
        self._ns_id_cntr += 1
        return ns_id

    def _deurify(self, uri: URIRef):
        if '#' in uri:
            ns, local_part = uri.split('#', 1)
            ns = ns + '#'
        else:
            ns, local_part = uri.rsplit('/', 1)
            ns = ns + '/'

        ns_id = self._ns_to_id.get(ns)

        if ns_id is None:
            ns_id = self._next_ns_id()
            self._ns_to_id[ns] = ns_id
            self._id_to_ns[ns_id] = ns

        return f'{ns_id}_{local_part}'

    def _urify(self, term):
        ns_id, local_part = term.split('_', 1)
        ns = self._id_to_ns[ns_id]
        uri = URIRef(ns + local_part)

        return uri

    def save_string_triplets(self, triplets_file_path: str):
        lines = []

        for s, p, o in self.rdf_graph:
            if isinstance(s, BNode):
                s_term = f'bnode_{str(s)}'
            else:
                s_term = self._deurify(s)

            p_term = self._deurify(p)

            if isinstance(o, Literal):
                o_term = f'lit_{o.value}'
            elif isinstance(o, BNode):
                o_term = f'bnode_{str(o)}'
            else:
                o_term = self._deurify(o)

            lines.append(f'{s_term}\t{p_term}\t{o_term}\n')

        with open(triplets_file_path, 'w') as out_file:
            out_file.writelines(lines)

    def get_embeddings_for_uris(
            self, embeddings: Embedding, entity_labels: dict) -> dict:

        uri_w_embedding = {}

        for e_id, embedding in enumerate(embeddings.weight):
            term = entity_labels[e_id]
            if term.startswith('lit_'):
                uri_w_embedding[Literal(term.split('_', 1)[1])] = embedding
            elif term.startswith('bnode_'):
                uri_w_embedding[BNode(term.split('_', 1)[1])] = embedding
            else:
                uri_w_embedding[self._urify(term)] = embedding

        return uri_w_embedding
