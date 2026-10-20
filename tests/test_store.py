import json
from datetime import date, timedelta

from german_tutor.store import INTERVALS, MAX_BOX, VocabStore

TODAY = date(2026, 1, 10)


def test_add_word_rejects_case_insensitive_duplicates(store: VocabStore):
    assert store.add_word("der Apfel", "apple", "Ich esse einen Apfel.")
    assert not store.add_word("DER APFEL", "apple", None)
    assert [w.german for w in store.all_words()] == ["der Apfel"]


def test_new_words_are_due_immediately(store: VocabStore):
    store.add_word("die Tür", "door", today=TODAY)
    assert [w.german for w in store.due_words(today=TODAY)] == ["die Tür"]


def test_correct_answers_promote_through_boxes_until_capped(store: VocabStore):
    store.add_word("die Birne", "pear", today=TODAY)
    word_id = store.all_words()[0].id

    day = TODAY
    for expected_box in [2, 3, 4, 5, 5]:
        result = store.record_review(word_id, correct=True, today=day)
        assert result.new_box == expected_box
        assert result.next_due == day + timedelta(days=INTERVALS[expected_box])
        day = result.next_due

    word = store.get_word(word_id)
    assert word.box == MAX_BOX and word.times_correct == 5


def test_wrong_answer_resets_to_box_one(store: VocabStore):
    store.add_word("der Bahnhof", "train station", today=TODAY)
    word_id = store.all_words()[0].id
    store.record_review(word_id, correct=True, today=TODAY)
    result = store.record_review(word_id, correct=False, today=TODAY)

    assert result.new_box == 1
    assert store.get_word(word_id).times_wrong == 1
    assert store.due_words(today=TODAY) == []
    assert len(store.due_words(today=TODAY + timedelta(days=1))) == 1


def test_due_words_orders_weakest_first(store: VocabStore):
    for g in ["eins", "zwei", "drei"]:
        store.add_word(g, g, today=TODAY)
    eins = store.all_words()[0].id
    store.record_review(eins, correct=True, today=TODAY - timedelta(days=5))  # box 2, overdue
    assert [w.german for w in store.due_words(today=TODAY)] == ["zwei", "drei", "eins"]


def test_record_review_unknown_id(store: VocabStore):
    assert store.record_review(999, correct=True) is None


def test_top_mistakes_groups_and_shows_latest_example(store: VocabStore):
    store.log_mistake("Perfekt with sein", "ich habe gegangen", "ich bin gegangen")
    store.log_mistake("dative case", "mit der Mann", "mit dem Mann")
    store.log_mistake("perfekt with sein ", "ich habe gefahren", "ich bin gefahren")

    top = store.top_mistakes()
    assert top[0] == ("perfekt with sein", 2, "ich habe gefahren", "ich bin gefahren")
    assert top[1][0] == "dative case"
    assert "perfekt with sein (2x)" in store.profile_summary()


def test_import_json(tmp_path, store: VocabStore):
    path = tmp_path / "words.json"
    path.write_text(json.dumps([
        {"german": "die Tür", "english": "door", "example": "Die Tür ist zu."},
        {"german": "die Tür", "english": "door"},
    ]), encoding="utf-8")
    assert store.import_json(path) == (1, 1)


def test_persists_to_disk(tmp_path):
    path = tmp_path / "nested" / "vocab.db"
    s = VocabStore(path)
    s.add_word("das Haus", "house")
    s.close()
    assert [w.german for w in VocabStore(path).all_words()] == ["das Haus"]
