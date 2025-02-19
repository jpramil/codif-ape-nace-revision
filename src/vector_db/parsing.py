def create_content_vdb(df):
    """Generate content for each row in the DataFrame."""

    def generate_content(row):
        sections = [
            f"# {row.code} : {row.label}",
            f"## Explications des activités incluses dans la sous-classe\n{row.notes}"
            if row.notes
            else None,
            f"## Liste d'exemples d'activités incluses dans la sous-classe\n{row.include}"
            if row.include
            else None,
            f"## Liste d'exemples d'activités non incluses dans la sous-classe\n{row.not_include}"
            if row.not_include
            else None,
        ]
        return "\n\n".join(filter(None, sections))

    df["content"] = df.apply(generate_content, axis=1)
    return df
