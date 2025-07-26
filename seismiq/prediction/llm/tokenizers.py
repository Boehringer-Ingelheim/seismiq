import abc
import re
from collections.abc import Iterable
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import rdchem  # type: ignore
from torch import Tensor

from seismiq.utils import rdkit_wrapper as wrdkit


class SmilesAtomTokenizer:
    # from Schwaller et. al. also used in deepchem
    SMI_REGEX_PATTERN = (
        r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|\+|\\\\\/|:|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    )
    TOKEN_PAD = ""
    TOKEN_START = "^"
    TOKEN_END = "&"
    TOKEN_UNK = "?"

    def __init__(self, smiles: Iterable[str] | None = None) -> None:
        self.token_to_id: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}

        self.add_token(self.TOKEN_PAD)
        self.add_token(self.TOKEN_START)
        self.add_token(self.TOKEN_END)
        self.add_token(self.TOKEN_UNK)
        for i in range(0, 10):
            self.add_token(str(i))
        for i in range(10, 100):
            self.add_token(f"%{i}")

        if smiles is not None:
            self.update_vocabulary(smiles)

    def add_token(self, tok: str) -> None:
        i = len(self.token_to_id)
        if tok not in self.token_to_id:
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok

    def update_vocabulary(self, smiles: Iterable[str]) -> None:
        for sm in sorted(set(smiles)):  # sort to guarantee deterministic behavior
            for tok in self.tokenize(sm):
                self.add_token(tok)

    @classmethod
    def tokenize(cls, smiles: str) -> list[str]:
        toks = re.findall(cls.SMI_REGEX_PATTERN, smiles)
        assert "" not in toks
        return toks

    @classmethod
    def detokenize(cls, tokens: list[str]) -> str:
        return "".join(tokens)

    def encode_tokens(self, tokens: list[str], dest: list[int] | None) -> list[int]:
        res = [self.token_to_id[tok] for tok in tokens]
        if dest is not None:
            dest.extend(res)
            return dest
        else:
            return res

    def decode_tokens(self, tokens: list[int], start_end_truncation: bool = True) -> list[str]:
        if start_end_truncation:
            tstart = self.token_to_id[self.TOKEN_START]
            tend = self.token_to_id[self.TOKEN_END]

            istart = iend = 0
            for i, t in enumerate(tokens):
                if t == tstart:
                    istart = i + 1
                elif t == tend:
                    iend = i
                    break

            tokens = tokens[istart:iend]

        return [self.id_to_token.get(tok, self.TOKEN_UNK) for tok in tokens]

    def encode(self, smiles: str) -> list[int]:
        res = [self.token_to_id[self.TOKEN_START]]
        self.encode_tokens(self.tokenize(smiles), res)
        res.append(self.token_to_id[self.TOKEN_END])
        return res

    def decode(self, tokens: list[int], start_end_truncation: bool = True) -> str:
        return self.detokenize(self.decode_tokens(tokens, start_end_truncation))

    @lru_cache
    def token_to_atom(self, token: int) -> tuple[str, int] | None:
        at = self.id_to_token.get(token)
        if not at:
            return None

        if at[0] == "[":
            assert at[-1] == "]"
            at = at[1:-1]

            # remove charge
            if at[-1].isdigit() or (at[-1] == "+" or at[-1] == "-"):  # e.g., "Co-3"
                at = at[:-1]

            # remove explicit Hs
            if at[-1] == "H":  # e.g., nH
                at = at[:-1]
            elif at[-1].isdigit():  # e.g., CH2
                assert at[-2] == "H"
                at = at[:-2]

        if not at:
            # this happens with '[H]' which we do not count
            return None

        # lower case are aromatic atoms
        if at[0].islower():
            at = at.capitalize()  # e.g., na -> Na

        if at and all(c.isalpha() for c in at):
            pt = Chem.GetPeriodicTable()  # type: ignore
            an = pt.GetAtomicNumber(at)
            return at, an
        else:
            return None


class TokenDencoder(abc.ABC):
    @abc.abstractmethod
    def encode(self, smiles: str) -> list[int]:
        pass

    @abc.abstractmethod
    def decode(self, tokens: list[int], return_mol: bool = True) -> rdchem.Mol | str | None:
        pass


class SmilesDencoder(TokenDencoder):
    def __init__(self, smiles_tokenizer: SmilesAtomTokenizer) -> None:
        self._smiles_tokenizer = smiles_tokenizer

    def encode(self, smiles: str) -> list[int]:
        return self._smiles_tokenizer.encode(smiles)

    def decode(self, tokens: list[int] | Tensor, return_mol: bool = True) -> wrdkit.Mol | str | None:
        if isinstance(tokens, Tensor):
            tokens = [int(t) for t in tokens.tolist()]

        sm = self._smiles_tokenizer.decode(tokens)
        if return_mol:
            if sm:
                return wrdkit.mol_from_smiles(sm)
            else:
                return None
        else:
            return sm
