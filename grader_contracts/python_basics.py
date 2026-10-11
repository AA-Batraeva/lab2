from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TextInput:
    value: str


@dataclass(frozen=True)
class PositiveIntegerInput:
    value: int


@dataclass(frozen=True)
class VectorPairInput:
    predicted: Sequence[int]
    expected: Sequence[int]

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    w = ('a', 'e', 'i', 'o', 'u')
    w1 = ('A', 'E', 'I', 'O', 'U')
    s = 0
    for i in text:
        if (i in w) or (i in w1):
            s = s + 1
    return s
    raise NotImplementedError  # TODO


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    s = len(text)
    for i in text:
        if text.count(i) > 1:
            return False
    return True
    raise NotImplementedError  # TODO





def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    s = 0
    while number > 0:
        if number % 2 == 1:
            s = s + 1
        number = number // 2
    return s
    raise NotImplementedError  # TODO


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    k = 0
    while number >= 10:
        p = 1
        k = k + 1
        for i in str(number):
            p = p * int(i)
        number = p
    return k
    raise NotImplementedError  # TODO


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    s = 0
    for i in range(len(predicted)):
        w = (predicted[i] - expected[i]) ** 2
        s = s + w
    sr = s / len(predicted)
    return sr
    raise NotImplementedError  # TODO


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    d = 2
    st = ''
    while number > 1:
        s = 0

        while number % d == 0:
            s = s + 1
            number = number // d
        if s > 1:
            st = st + '(' + str(d) + '**' + str(s) + ')'
        elif s == 0:
            st = st
        elif s == 1:
            st = st + '(' + str(d) + ')'
        d = d + 1
    return st
    raise NotImplementedError  # TODO


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    k = 0
    s = 0
    while cube_count - s > 0:
        k = k + 1
        s = s + k * k
    if cube_count - s == 0:
        return k
    else:
        return "It is impossible"
    raise NotImplementedError  # TODO

def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    st = str(number)
    l = len(st)

    mid = l // 2
    if l % 2 == 0:
         w = st[:mid - 1]
         p = st[mid + 1:]
    else:
         w = st[:mid]
         p = st[mid + 1:]

    s = sum(int(i) for i in w)
    s2 = sum(int(i) for i in p)

    return s == s2
    raise NotImplementedError  # TODO
