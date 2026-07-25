"""
Os cinco clientes fixos usados na avaliação (Etapa 4) e na demo (Etapa 8).

Vivem aqui, e não dentro de um dos dois, porque os dois precisam exatamente dos mesmos
perfis: o golden set congela a recomendação de cada um, e a página de demo os oferece como
botões. Se divergissem, a demo mostraria um caso que nenhum teste protege.

Cada persona traz uma `seed`. Com ela a recomendação é reproduzível (ver README, “Reprodutibilidade”) — é o que
permite ensaiar a apresentação sabendo o que a API vai responder.

Ressalva que precisa acompanhar estes dados onde quer que apareçam: a política é
**não-contextual** (ver README, “Escolhas de design”). Os perfis descrevem a variedade da base e alimentam o log da
API, mas não alteram a oferta escolhida. O que distingue as respostas é a seed.
"""

from __future__ import annotations

from typing import Any, Dict, List

CLIENT_PERSONAS: List[Dict[str, Any]] = [
    {
        "case_id": "previous_converter",
        "descricao": "Já converteu em campanha anterior — segmento de maior conversão real (65,1%).",
        "seed": 101,
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
        "seed": 202,
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
        "seed": 303,
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
        "seed": 404,
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
            "previous": 0,
            "poutcome": "nonexistent",
        },
    },
    {
        "case_id": "low_engagement",
        "descricao": "Nunca contactado antes — o caso mais frio, conversão real de 3,7%.",
        "seed": 505,
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
