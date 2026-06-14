# V2 Architecture & Span Grouping Diagrams

These Mermaid diagrams can be rendered with any Mermaid-compatible tool
(VS Code preview, GitHub, mermaid-cli → PNG/SVG/PDF).

---

## Diagram 1: V1 vs V2 Architecture Comparison

```mermaid
flowchart TB
    subgraph Input["Document Input"]
        A["OCR Words + Bounding Boxes"]
    end

    subgraph Shared["Shared Backbone"]
        B["LayoutLMv3 Encoder"]
        C["Entity Classifier<br/>(Key / Value / Other)"]
    end

    A --> B --> C

    subgraph V1["V1: Token-Level Linking"]
        direction TB
        D1["Individual Tokens<br/>~100 keys × ~100 values"]
        E1["Token-Level Biaffine Scorer<br/>100×100 = 10,000 pairs"]
        F1["BCE Loss<br/>~450:1 neg:pos ratio"]
        G1["Result: Link F1 ≈ 0.008"]
        D1 --> E1 --> F1 --> G1
    end

    subgraph V2["V2: Span-Level Linking"]
        direction TB
        D2["Span Grouping<br/>Contiguous same-label → spans"]
        D2a["~15 key spans × ~15 value spans"]
        E2["Mean-Pool + Union BBox<br/>Span Representations"]
        F2["Span Biaffine Scorer<br/>15×15 = 225 pairs"]
        G2["BCE Loss<br/>~5:1 neg:pos ratio"]
        D2 --> D2a --> E2 --> F2 --> G2
    end

    C --> D1
    C --> D2

    style V1 fill:#ffcccc,stroke:#cc0000
    style V2 fill:#ccffcc,stroke:#00cc00
    style G1 fill:#ff9999,stroke:#cc0000
    style G2 fill:#99ff99,stroke:#00cc00
```

---

## Diagram 2: Span Grouping Detail

```mermaid
flowchart LR
    subgraph Tokens["Token Sequence (Entity Labels)"]
        T1["Invoice"]:::other
        T2["Date"]:::key
        T3[":"]:::key
        T4["01"]:::val
        T5["/"]:::val
        T6["15"]:::val
        T7["/"]:::val
        T8["2024"]:::val
        T9["Total"]:::key
        T10["Amount"]:::key
        T11["$"]:::val
        T12["1,500"]:::val
    end

    subgraph Spans["Grouped Spans"]
        S1["Key Span 1<br/>'Date :'<br/>tokens 2-3"]:::keyspan
        S2["Value Span 1<br/>'01/15/2024'<br/>tokens 4-8"]:::valspan
        S3["Key Span 2<br/>'Total Amount'<br/>tokens 9-10"]:::keyspan
        S4["Value Span 2<br/>'$ 1,500'<br/>tokens 11-12"]:::valspan
    end

    subgraph Matrix["Span Link Matrix (2×2)"]
        M["| | Val1 | Val2 |<br/>|---|---|---|<br/>| Key1 | ✓ | ✗ |<br/>| Key2 | ✗ | ✓ |"]
    end

    T2 & T3 --> S1
    T4 & T5 & T6 & T7 & T8 --> S2
    T9 & T10 --> S3
    T11 & T12 --> S4
    S1 & S2 & S3 & S4 --> Matrix

    classDef other fill:#eeeeee,stroke:#999999
    classDef key fill:#ff9999,stroke:#cc0000
    classDef val fill:#9999ff,stroke:#0000cc
    classDef keyspan fill:#ff6666,stroke:#cc0000,color:#fff
    classDef valspan fill:#6666ff,stroke:#0000cc,color:#fff
```

---

## Diagram 3: End-to-End Pipeline

```mermaid
flowchart LR
    A["Raw Document<br/>(PDF/Image)"] --> B["OCR<br/>(Tesseract)"]
    B --> C["LMDX Text Format<br/>(words + bboxes)"]
    C --> D["LayoutLMv3<br/>Encoder"]
    D --> E["Entity Classifier<br/>(Key/Value/Other)"]
    E --> F["Span Grouper<br/>(contiguous merge)"]
    F --> G["Biaffine Linker<br/>(span-level)"]
    G --> H["KVP Pairs<br/>(key_text → value_text)"]

    style D fill:#e6f3ff,stroke:#0066cc
    style E fill:#fff3e6,stroke:#cc6600
    style F fill:#e6ffe6,stroke:#00cc00
    style G fill:#ffe6e6,stroke:#cc0000
```
