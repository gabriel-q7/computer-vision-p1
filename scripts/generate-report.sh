#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR="docs"
OUTPUT_MD="$OUTPUT_DIR/final-report.md"
OUTPUT_PDF="$OUTPUT_DIR/gabriel_aguiar_aquino-deep-learning-and-computer-vision.pdf"

REPORT_TITLE="Deep Learning and Computer Vision"
REPORT_SUBTITLE="Relatório das Atividades 1 a 4"
REPORT_AUTHOR="Gabriel Aguiar Aquino"

# Paths are relative to the repository root. Their order is preserved.
REPORT_SOURCES=(
    "docs/ai-use-disclosure.md"
    "docs/introducao.md"

    "atividade-1-projeto-livre/RELATORIO.md"
    "atividade-2-clip-publicidade/RELATORIO.md"
    "atividade-3-cnn-pre-treinada/RELATORIO.md"
    "atividade-4-estudos-de-caso/RELATORIO.md"

    "docs/referencias.md"
)

# Extracts the report figures from the notebook outputs before building the PDF.
FIGURE_EXTRACTOR="scripts/extract_notebook_figures.py"

# ============================================================
# Implementation
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR_ABS="$REPO_ROOT/$OUTPUT_DIR"
OUTPUT_MD_ABS="$REPO_ROOT/$OUTPUT_MD"
OUTPUT_PDF_ABS="$REPO_ROOT/$OUTPUT_PDF"

info() {
    printf '[INFO] %s\n' "$*"
}

warn() {
    printf '[WARNING] %s\n' "$*" >&2
}

error() {
    printf '[ERROR] %s\n' "$*" >&2
}

display_path() {
    local path="$1"
    if [[ "$path" == "$REPO_ROOT/"* ]]; then
        printf '%s' "${path#"$REPO_ROOT/"}"
    else
        printf '%s' "$path"
    fi
}

yaml_escape() {
    local value="$1"
    value="${value//\\/\\\\}"
    value="${value//\"/\\\"}"
    printf '%s' "$value"
}

# Report date in Brazilian Portuguese, independent of the system locale.
report_date() {
    local months=(janeiro fevereiro março abril maio junho julho agosto setembro outubro novembro dezembro)
    local day month year
    day="$(date +%-d)"
    month="$(date +%-m)"
    year="$(date +%Y)"
    printf '%s de %s de %s' "$day" "${months[$((month - 1))]}" "$year"
}

validate_dependencies() {
    local dependency
    local missing=0

    info "Validating dependencies..."
    for dependency in bash pandoc xelatex python3; do
        if command -v "$dependency" >/dev/null 2>&1; then
            info "Dependency available: $dependency"
        else
            error "Required dependency not found: $dependency"
            missing=1
        fi
    done

    if ((missing)); then
        error "Install the missing tool(s) and run the script again. No packages were installed automatically."
        return 1
    fi
}

extract_figures() {
    info "Extracting figures from notebook outputs..."
    if ! python3 "$REPO_ROOT/$FIGURE_EXTRACTOR"; then
        error "Figure extraction failed. Check the cell indices in $FIGURE_EXTRACTOR."
        return 1
    fi
}

validate_sources() {
    local source
    local source_abs
    local missing=0

    info "Validating Markdown sources..."
    for source in "${REPORT_SOURCES[@]}"; do
        source_abs="$REPO_ROOT/$source"
        if [[ ! -f "$source_abs" ]]; then
            error "Markdown source not found:"
            printf '%s\n' "$source" >&2
            missing=1
        fi
    done

    if ((missing)); then
        error "Source validation failed. No report was generated."
        return 1
    fi
}

warn_if_image_missing() {
    local source_abs="$1"
    local image_ref="$2"
    local source_dir

    [[ -z "$image_ref" ]] && return 0
    case "$image_ref" in
        http://*|https://*|data:*|mailto:*|\#*) return 0 ;;
    esac

    source_dir="$(dirname "$source_abs")"
    if [[ "$image_ref" == /* ]]; then
        [[ -f "$image_ref" ]] || warn "Image referenced by $(display_path "$source_abs") was not found: $image_ref"
    elif [[ ! -f "$source_dir/$image_ref" && ! -f "$REPO_ROOT/$image_ref" ]]; then
        warn "Image referenced by $(display_path "$source_abs") was not found: $image_ref"
    fi
}

validate_image_references() {
    local source
    local source_abs
    local match
    local image_ref

    info "Checking local image references..."
    for source in "${REPORT_SOURCES[@]}"; do
        source_abs="$REPO_ROOT/$source"

        while IFS= read -r match; do
            [[ -z "$match" ]] && continue
            image_ref="${match#*](}"
            image_ref="${image_ref%)}"
            if [[ "$image_ref" == \<*\> ]]; then
                image_ref="${image_ref#<}"
                image_ref="${image_ref%>}"
            else
                image_ref="${image_ref%%[[:space:]]*}"
            fi
            warn_if_image_missing "$source_abs" "$image_ref"
        done < <(grep -oE '!\[[^]]*\]\([^)]*\)' "$source_abs" || true)

        while IFS= read -r match; do
            [[ -z "$match" ]] && continue
            image_ref="${match#*src=\"}"
            image_ref="${image_ref%%\"*}"
            warn_if_image_missing "$source_abs" "$image_ref"
        done < <(grep -oE '<img[^>]+src="[^"]+"' "$source_abs" || true)
    done
}

# Each activity keeps its images in its own "figures/" directory. When the sources are
# concatenated, those relative paths are rewritten relative to the repository root so that
# files with the same name in different activities can never be confused.
rewrite_figure_paths() {
    local source_dir_rel="$1"
    if [[ "$source_dir_rel" == "." ]]; then
        command cat
    else
        sed -E "s#(!\[[^]]*\]\()(figures/)#\1${source_dir_rel}/\2#g"
    fi
}

generate_markdown() {
    local source
    local source_abs
    local temp_md

    info "Generating consolidated Markdown..."
    mkdir -p "$OUTPUT_DIR_ABS"
    temp_md="$(mktemp "$OUTPUT_DIR_ABS/.final-report.md.XXXXXX")"

    {
        printf '%s\n' '---'
        printf 'title: "%s"\n' "$(yaml_escape "$REPORT_TITLE")"
        printf 'subtitle: "%s"\n' "$(yaml_escape "$REPORT_SUBTITLE")"
        printf 'author: "%s"\n' "$(yaml_escape "$REPORT_AUTHOR")"
        printf 'date: "%s"\n' "$(yaml_escape "$(report_date)")"
        printf '%s\n' 'lang: pt-BR' 'geometry: margin=2.5cm' 'toc: true' 'numbersections: true'
        printf '%s\n' 'header-includes:' \
            '  - \usepackage{float}' \
            '  - \floatplacement{figure}{H}' \
            '---' '' '\newpage'

        for source in "${REPORT_SOURCES[@]}"; do
            source_abs="$REPO_ROOT/$source"
            info "Adding: $source" >&2
            printf '\n\n<!-- Source: %s -->\n\n' "$source"
            rewrite_figure_paths "$(dirname "$source")" <"$source_abs"
            printf '\n\n\\newpage\n'
        done
    } >"$temp_md"

    mv "$temp_md" "$OUTPUT_MD_ABS"
}

generate_pdf() {
    local temp_pdf

    info "Generating PDF..."
    temp_pdf="$OUTPUT_DIR_ABS/.final-report.$$.pdf"
    command rm -f "$OUTPUT_PDF_ABS"

    if ! (
        cd "$REPO_ROOT"
        pandoc "$OUTPUT_MD_ABS" \
            --from=markdown+pipe_tables+fenced_code_blocks \
            --pdf-engine=xelatex \
            --resource-path="$REPO_ROOT" \
            --toc \
            --number-sections \
            --syntax-highlighting=tango \
            --variable=papersize:a4 \
            --variable=fontsize:11pt \
            --variable=babel-lang:portuguese-brazil \
            --variable=mainfont:"DejaVu Serif" \
            --variable=sansfont:"DejaVu Sans" \
            --variable=monofont:"DejaVu Sans Mono" \
            --variable=colorlinks:true \
            --variable=linkcolor:blue \
            --variable=urlcolor:blue \
            --output="$temp_pdf"
    ); then
        command rm -f "$temp_pdf"
        error "PDF generation failed."
        error "The Markdown report was generated successfully:"
        printf '%s\n' "$OUTPUT_MD" >&2
        return 1
    fi

    mv "$temp_pdf" "$OUTPUT_PDF_ABS"
}

main() {
    info "Repository root: $REPO_ROOT"
    validate_dependencies
    extract_figures
    validate_sources
    validate_image_references
    generate_markdown
    generate_pdf

    info "Report generated successfully."
    printf '\nMarkdown:\n  %s\n\nPDF:\n  %s\n' "$OUTPUT_MD" "$OUTPUT_PDF"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
