import requests
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# =============================
# CONFIG
# =============================

API_BASE = "https://movie-recommendation-system-wh20.onrender.com" or "http://127.0.0.1:8000"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# =============================
# STYLES
# =============================
st.markdown(
    """
<style>
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2.5rem;
    max-width: 1400px;
}

/* Card */
.card {
    border-radius: 18px;
    padding: 12px;
    background: #ffffff;
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
    transition: transform .15s ease, box-shadow .15s ease;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.10);
}

/* Poster */
.poster-wrapper {
    width: 100%;
    aspect-ratio: 2 / 3;
    border-radius: 14px;
    overflow: hidden;
    background: #f3f4f6;
}

.poster-wrapper img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

/* Title */
.movie-title {
    font-size: 0.92rem;
    font-weight: 600;
    margin-top: 8px;

    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;

    overflow: hidden;
    text-overflow: ellipsis;
    line-height: 1.2rem;
    min-height: 2.4rem;
}

.small-muted {
    color: #6b7280;
    font-size: 0.9rem;
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# STATE
# =============================
if "view" not in st.session_state:
    st.session_state.view = "home"
if "selected_tmdb_id" not in st.session_state:
    st.session_state.selected_tmdb_id = None

# =============================
# NAVIGATION
# =============================
def goto_home():
    st.session_state.view = "home"
    st.session_state.selected_tmdb_id = None
    st.rerun()


def goto_details(tmdb_id: int):
    st.session_state.view = "details"
    st.session_state.selected_tmdb_id = int(tmdb_id)
    st.rerun()


# =============================
# API HELPER
# =============================
@st.cache_data(ttl=120, show_spinner=False)
def api_get_json(path: str, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=25)
        if r.status_code >= 400:
            return None
        return r.json()
    except Exception:
        return None

# =============================
# UI HELPERS
# =============================
def poster_grid(cards, cols=6, key_prefix="grid"):
    if not cards:
        st.info("No movies found.")
        return

    rows = (len(cards) + cols - 1) // cols
    idx = 0

    for _ in range(rows):
        colset = st.columns(cols, gap="medium")
        for col in colset:
            if idx >= len(cards):
                return

            m = cards[idx]
            idx += 1

            tmdb_id = m.get("tmdb_id")
            title = m.get("title", "Untitled")
            poster = m.get("poster_url")

            with col:
                st.markdown("<div class='card'>", unsafe_allow_html=True)

                if poster:
                    st.markdown(
                        f"""
                        <div class="poster-wrapper">
                            <img src="{poster}" />
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        """
                        <div class="poster-wrapper"
                             style="display:flex;align-items:center;justify-content:center;
                                    font-size:0.85rem;color:#9ca3af;">
                            No poster
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"<div class='movie-title' title='{title}'>{title}</div>",
                    unsafe_allow_html=True,
                )

                if tmdb_id:
                    if st.button(
                        "View details",
                        key=f"{key_prefix}_{tmdb_id}",
                        use_container_width=True,
                    ):
                        goto_details(tmdb_id)

                st.markdown("</div>", unsafe_allow_html=True)

def tfidf_cards(items):
    out = []
    for x in items or []:
        tmdb = x.get("tmdb") or {}
        if tmdb.get("tmdb_id"):
            out.append(
                {
                    "tmdb_id": tmdb["tmdb_id"],
                    "title": tmdb.get("title"),
                    "poster_url": tmdb.get("poster_url"),
                }
            )
    return out

# =============================
# SIDEBAR
# =============================
with st.sidebar:
    st.markdown("## 🎬 Menu")

    if st.button("🏠 Home", use_container_width=True):
        goto_home()

    st.markdown("---")

    home_category = st.selectbox(
        "Home Category",
        ["trending", "popular", "top_rated", "now_playing", "upcoming"],
    )

    grid_cols = st.slider("Grid Columns", 4, 8, 6)

# =============================
# HEADER
# =============================
st.title("🎬 Movie Recommender")
st.markdown(
    "<div class='small-muted'>Search → explore → details → recommendations</div>",
    unsafe_allow_html=True,
)
st.divider()

# ==========================================================
# HOME VIEW
# ==========================================================
if st.session_state.view == "home":
    query = st.text_input("🔍 Search movies", placeholder="Avengers, Batman, Love…")

    if query.strip():
        data = api_get_json("/tmdb/search", {"query": query.strip()})

        if not data:
            st.error("Search failed.")
        else:
            results = sorted(
                data.get("results", []),
                key=lambda x: x.get("popularity", 0),
                reverse=True,
            )

            cards = [
                {
                    "tmdb_id": m["id"],
                    "title": m.get("title"),
                    "poster_url": f"{TMDB_IMG}{m['poster_path']}"
                    if m.get("poster_path")
                    else None,
                }
                for m in results[:24]
            ]

            st.markdown("### 🔎 Results")
            poster_grid(cards, cols=grid_cols, key_prefix="search")

    else:
        st.markdown(f"### 🏠 {home_category.replace('_',' ').title()} Movies")
        home_cards = api_get_json("/home", {"category": home_category, "limit": 24})
        poster_grid(home_cards or [], cols=grid_cols, key_prefix="home")

# ==========================================================
# DETAILS VIEW
# ==========================================================
elif st.session_state.view == "details":
    tmdb_id = st.session_state.selected_tmdb_id

    if st.button("← Back to Home"):
        goto_home()

    data = api_get_json(f"/movie/id/{tmdb_id}")
    if not data:
        st.error("Failed to load movie.")
        st.stop()

    left, right = st.columns([1, 2.5], gap="large")

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="poster-wrapper">
                <img src="{data.get('poster_url')}" />
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"## {data.get('title')}")
        st.markdown(
            f"<div class='small-muted'>Release: {data.get('release_date','-')}</div>",
            unsafe_allow_html=True,
        )
        genres = ", ".join(g["name"] for g in data.get("genres", [])) or "-"
        st.markdown(
            f"<div class='small-muted'>Genres: {genres}</div>",
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.write(data.get("overview", "No overview available."))
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("### ✅ Recommendations")

    bundle = api_get_json(
        "/movie/search",
        {
            "query": data.get("title"),
            "tfidf_top_n": 12,
            "genre_limit": 12,
        },
    )

    if bundle:
        st.markdown("#### 🔎 Similar Movies")
        poster_grid(
            tfidf_cards(bundle.get("tfidf_recommendations")),
            cols=grid_cols,
            key_prefix="tfidf",
        )

        st.markdown("#### 🎭 More Like This")
        poster_grid(
            bundle.get("genre_recommendations"),
            cols=grid_cols,
            key_prefix="genre",
        )
