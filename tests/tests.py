import pytest
from pytest_mock import mocker
from src.ingestion import get_events

#Test de la fonction de récupération des events
def test_get_events_success(mocker):
    #On simule une réponse réussie
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [{"title_fr": "Test Event", "location_city": "Paris"}]
    }

    mocker.patch("requests.get", return_value=mock_response)

    params = {"limit": 1}
    results = get_events(params)

    assert len(results) == 1
    assert results[0]["title_fr"] == "Test Event"

def test_get_events_failure(mocker):
    #ON simule un erreur 404
    mock_response = mocker.Mock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"

    mocker.patch("requests.get", return_value=mock_response)

    with pytest.raises(Exception):
        get_events()