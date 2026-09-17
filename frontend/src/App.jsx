import { useState, useEffect } from "react"

const colors = {
  bg: '#1A1D21',
  surface: '#22262B',
  border: '#2C3036',
  red: '#E2453A',
  blue: '#3E8EDE',
  ink: '#EDEBE3',
  inkMuted: '#9A9D9F',
};

function ConfidenceBar({ percentLeft }) {
  return (
    <div style={{ display: 'flex', height: '6px', width: '100%', background: '#33383F', marginBottom: '12px' }}>
      <div style={{ width: `${percentLeft}%`, background: colors.red }} />
      <div style={{ width: `${100 - percentLeft}%`, background: colors.blue }} />
    </div>
  );
}

function HeadToHead() {
  const [fighterA, setFighterA] = useState("");
  const [fighterB, setFighterB] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false)

  function handleSubmit(e) {
    e.preventDefault();
    setLoading(true)
    setResult(null)

    fetch("http://127.0.0.1:8000/predict", {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fighter_a: fighterA, fighter_b: fighterB }),
    })
      .then(response => response.json())
      .then(data => {
        setResult(data);
        setLoading(false);
      })
  }

  const inputStyle = {
    flex: 1,
    minWidth: '160px',
    background: colors.surface,
    border: `1px solid ${colors.border}`,
    color: colors.ink,
    padding: '10px 14px',
    fontSize: '14px',
    fontFamily: 'Inter, sans-serif',
  };

  return (
    <div style={{ maxWidth: '600px', margin: '0 auto' }}>
      <form onSubmit={handleSubmit} style={{ display: 'flex', gap: '12px', flexWrap: 'wrap', alignItems: 'center' }}>
        <input
          value={fighterA}
          onChange={(e) => setFighterA(e.target.value)}
          placeholder="Fighter A"
          style={inputStyle}
        />
        <span style={{ color: colors.inkMuted, fontSize: '12px' }}>vs</span>
        <input
          value={fighterB}
          onChange={(e) => setFighterB(e.target.value)}
          placeholder="Fighter B"
          style={inputStyle}
        />
        <button
          type="submit"
          disabled={loading}
          style={{
            background: colors.ink,
            color: colors.bg,
            border: 'none',
            padding: '10px 20px',
            fontSize: '14px',
            fontWeight: 600,
            cursor: loading ? 'default' : 'pointer',
            fontFamily: 'Inter, sans-serif',
          }}
        >
         {loading ? 'Predicting...': 'Predict'}
        </button>
      </form>

      {result && (
        <div style={{ background: colors.surface, border: `1px solid ${colors.border}`, padding: '18px 20px', marginTop: '16px' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
            <div style={{ fontFamily: 'Oswald, sans-serif', fontSize: '19px', color: colors.red, textAlign: 'left' }}>
              {result.fighter_a}
            </div>
            <div style={{ color: colors.inkMuted, fontSize: '12px' }}>vs</div>
            <div style={{ fontFamily: 'Oswald, sans-serif', fontSize: '19px', color: colors.blue, textAlign: 'right' }}>
              {result.fighter_b}
            </div>
          </div>
          <ConfidenceBar percentLeft={Math.round(result.fighter_a_win_prob * 100)} />
          <p style={{ color: colors.inkMuted, fontSize: '13px', margin: 0 }}>
            {result.fighter_a}: {Math.round(result.fighter_a_win_prob * 100)}%
            &nbsp;·&nbsp;
            {result.fighter_b}: {Math.round(result.fighter_b_win_prob * 100)}%
          </p>
        </div>
      )}
    </div>
  );
}

function FightCard({ fighterA, fighterB, probA, probB, isArchive, correct, actualWinner, method, event, date }) {
  const percentLeft = Math.round(probA * 100);
  const pickedWinner = probA >= probB ? fighterA : fighterB;

  return (
    <div style={{ background: colors.surface, border: `1px solid ${colors.border}`, padding: '18px 20px', marginBottom: '14px' }}>
      <div style={{ fontSize: '12px', color: colors.inkMuted, marginBottom: '12px' }}>
        {event} · {new Date(date).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', alignItems: 'center', gap: '12px', marginBottom: '10px' }}>
        <div style={{ fontFamily: 'Oswald, sans-serif', fontSize: '19px', color: colors.red, textAlign: 'left' }}>
          {fighterA}
        </div>
        <div style={{ color: colors.inkMuted, fontSize: '12px' }}>vs</div>
        <div style={{ fontFamily: 'Oswald, sans-serif', fontSize: '19px', color: colors.blue, textAlign: 'right' }}>
          {fighterB}
        </div>
      </div>

      <ConfidenceBar percentLeft={percentLeft} />

      <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
        <p style={{ color: colors.inkMuted, fontSize: '13px', margin: 0 }}>
          Model pick: <strong style={{ color: colors.ink }}>{pickedWinner}</strong>
        </p>

        {isArchive && (
          <p style={{
            fontSize: '12px',
            margin: 0,
            padding: '4px 10px',
            border: `1px solid ${correct ? '#3A5A40' : '#5A3A38'}`,
            color: correct ? '#7CC48A' : '#E2837A',
          }}>
            {correct ? 'Correct' : 'Missed'} · {actualWinner} won by {method}
          </p>
        )}
      </div>
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

  function navButtonStyle(isActive) {
    return {
      background: 'transparent',
      border: `1px solid ${isActive ? colors.ink : colors.border}`,
      color: isActive ? colors.ink : colors.inkMuted,
      padding: '8px 18px',
      fontSize: '14px',
      cursor: 'pointer',
      fontFamily: 'Inter, sans-serif',
    };
  }

  return (
    <div style={{ background: colors.bg, color: colors.ink, minHeight: '100vh', padding: '32px 24px 64px', fontFamily: 'Inter, sans-serif' }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=Inter:wght@400;500;600&display=swap');
        * { box-sizing: border-box; }
        input::placeholder { color: ${colors.inkMuted}; }
      `}</style>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '16px', marginBottom: '28px' }}>
        <div style={{ fontFamily: 'Oswald, sans-serif', fontWeight: 700, fontSize: '22px', letterSpacing: '0.03em' }}>
          FIGHT CARD PREDICTIONS
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <button onClick={() => setTab("upcoming")} style={navButtonStyle(tab === "upcoming")}>Upcoming</button>
          <button onClick={() => setTab("archive")} style={navButtonStyle(tab === "archive")}>Archive</button>
          <button onClick={() => setTab("h2h")} style={navButtonStyle(tab === "h2h")}>Head-to-Head</button>
        </div>
      </div>

      {tab === "h2h" ? (
        <HeadToHead />
      ) : (
        <div style={{ maxWidth: '720px', margin: '0 auto' }}>
          {fightsToShow.map(fight => (
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
              event={fight.event}
              date={fight.date}
            />
          ))}
        </div>
      )}
    </div>
  )
}

export default App;
