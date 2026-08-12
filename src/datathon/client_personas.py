"""
Os cinco clientes fixos usados na avaliação (Etapa 4) e na demo (Etapa 8).

Vivem aqui, e não dentro de um dos dois, porque os dois precisam exatamente dos mesmos
perfis: o golden set congela a recomendação de cada um, e a página de demo os oferece como
botões. Se divergissem, a demo mostraria um caso que nenhum teste protege.

Todas as personas compartilham a mesma `GOLDEN_SEED` (ver README, "Reprodutibilidade"). De
propósito: a seed não é parâmetro do modelo, é só o sorteio Thompson dentro do segmento — se
cada persona tivesse uma seed diferente, uma eventual diferença de oferta entre duas
personas poderia ser por causa do sorteio, não do perfil. Com a seed fixa e igual para
todas, a única variável que resta é o segmento — o argumento de personalização fica limpo.

A política é contextual: os perfis determinam o segmento cujo posterior participa da
decisão. A seed controla apenas o sorteio reprodutível dentro desse segmento.
"""

from __future__ import annotations

from typing import Any, Dict, List

GOLDEN_SEED = 42
"""
Seed única, compartilhada por todas as personas (e pelo modo "Personalizado" da demo).

Não há motivo técnico para variar por caso — variar só convidaria a confundir "mudou de
oferta por causa do perfil" com "mudou por causa do sorteio". Ver docstring do módulo.
"""

CLIENT_PERSONAS: List[Dict[str, Any]] = [
    {
        "case_id": "previous_converter",
        "descricao": "Já converteu em campanha anterior — segmento de maior conversão real (65,1%).",
        "seed": GOLDEN_SEED,
        "client": {
            "age": 47,
            "job": "management",
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "day_of_week": "mon",
            "campaign": 1,
            "previous": 2,
            "poutcome": "success",
        },
    },
    {
        "case_id": "student_digital",
        "descricao": "Estudante em canal digital, em alta temporada (março).",
        "seed": GOLDEN_SEED,
        "client": {
            "age": 22,
            "job": "student",
            "marital": "single",
            "education": "university.degree",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "cellular",
            "month": "mar",
            "day_of_week": "tue",
            "campaign": 1,
            "previous": 0,
            "poutcome": "nonexistent",
        },
    },
    {
        "case_id": "retired",
        "descricao": "Aposentado — perfil conservador, conversão real de 17,6%.",
        "seed": GOLDEN_SEED,
        "client": {
            "age": 68,
            "job": "retired",
            "marital": "married",
            "education": "basic.4y",
            "default": "no",
            "housing": "no",
            "loan": "no",
            "contact": "telephone",
            "month": "oct",
            "day_of_week": "wed",
            "campaign": 2,
            "previous": 0,
            "poutcome": "nonexistent",
        },
    },
    {
        "case_id": "digital_channel_massa",
        "descricao": "Cliente típico da maior fatia da base (54% em canal digital).",
        "seed": GOLDEN_SEED,
        "client": {
            "age": 38,
            "job": "admin.",
            "marital": "married",
            "education": "high.school",
            "default": "no",
            "housing": "yes",
            "loan": "yes",
            "contact": "cellular",
            "month": "jul",
            "day_of_week": "thu",
            "campaign": 3,
            "previous": 1,
            "poutcome": "nonexistent",
        },
    },
    {
        "case_id": "low_engagement",
        "descricao": "Nunca contactado antes — o caso mais frio, conversão real de 3,7%.",
        "seed": GOLDEN_SEED,
        "client": {
            "age": 31,
            "job": "blue-collar",
            "marital": "single",
            "education": "basic.9y",
            "default": "unknown",
            "housing": "no",
            "loan": "no",
            "contact": "telephone",
            "month": "jun",
            "day_of_week": "fri",
            "campaign": 1,
            "previous": 0,
            "poutcome": "nonexistent",
        },
    },
]
