import { useState, useEffect } from "react"


function HeadToHead() {
  const [fighterA, setFighterA] = useState("");
  const [fighterB, setFighterB] = useState("");
  const [result, setResult] = useState(null);

  function handleSubmit(e) {
    e.preventDefault();

    fetch("http://127.0.0.1:8000/predict", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({fighter_a:fighterA, fighter_b:fighterB}),
    })
      .then(response => response.json())
      .then(data => setResult(data))
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input
          value={fighterA}
          onChange={(e) => setFighterA(e.target.value)}
          placeholder="Fighter A" />
        <input
          value={fighterB}
          onChange={(e) => setFighterB(e.target.value)}
          placeholder="Fighter B"
        />
        <button type="submit">Predict</button>
      </form>


      {result && (
        <div>
          <p>{result.fighter_a}: {Math.round(result.fighter_a_win_prob * 100)}</p>
          <p>{result.fighter_b}: {Math.round(result.fighter_b_win_prob * 100)}</p>
        </div>
      )}
    </div>
  )
}


function ConfidenceBar({ percentLeft }) {
  return (
    <div style={{ display: 'flex', height: '8px', width: '100%', background: '#ddd' }} >
      <div style={{ width: `${percentLeft}%`, background: 'red' }} />
      <div style={{ width: `${100 - percentLeft}%`, background: 'blue' }} />
    </div>
  );
}

function FightCard({fighterA, fighterB, probA, probB, isArchive, correct, actualWinner, method}) {
  const percentLeft = Math.round(probA * 100);
  const pickedWinner = probA >= probB ? fighterA : fighterB;

  return (
    <div style={{ border: '1px solid #ccc', padding: '12px', marginBottom: '10px' }}>
      <h3>{fighterA} vs {fighterB}</h3>
      <ConfidenceBar percentLeft={percentLeft} />
      <p>Model picks: {pickedWinner}</p>

      {isArchive && (
        <p style={{ color: correct ? 'green' : 'red' }}>
          {correct ? 'Correct' : 'Missed'} - {actualWinner} won by {method}
        </p>
      )}
    </div>
  );
}

function App() {
  const [tab, setTab] = useState("upcoming")
  const [upcoming, setUpcoming] = useState([]);
  const [archive, setArchive] = useState([])
  useEffect(() => {
    fetch('http://127.0.0.1:8000/predictions/upcoming')
         .then(response => response.json())
         .then(data => setUpcoming(data));
    fetch('http://127.0.0.1:8000/predictions/archive')
         .then(response => response.json())
         .then(data => setArchive(data));
  }, []);

  const fightsToShow = tab === "upcoming" ? upcoming : archive;


  return (
    <div>
      <h1>Fight Predictions</h1>
      <button onClick={() => setTab("upcoming")}>Upcoming</button>
      <button onClick={() => setTab("archive")}>Archive</button>
      <button onClick={() => setTab("h2h")}>Head-to-Head</button>
      {tab === "h2h" ? (
        <HeadToHead />
      ) : (
        fightsToShow.map(fight => (
          <FightCard
            key={fight.fighter_a + fight.fighter_b + fight.date}
            fighterA={fight.fighter_a}
            fighterB={fight.fighter_b}
            probA={fight.fighter_a_win_prob}
            probB={fight.fighter_b_win_prob}
            isArchive={tab === "archive"}
            correct={fight.correct}
            actualWinner={fight.actual_winner}
            method={fight.method}
          />
        ))
      )}
    </div>
  )
}


export default App;
