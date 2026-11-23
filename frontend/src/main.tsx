import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import { apiUrl, request, type Correlations, type Explanation, type Feature, type GlobalShap, type Model, type Prediction, type Summary } from './api'
import './style.css'

const pages = ['Overview', 'Assessment', 'Explainability', 'Dataset', 'Performance', 'Methodology'] as const
type Page = typeof pages[number]
type Example = { prediction: Prediction; explanation?: Explanation }
const number = new Intl.NumberFormat('en', { maximumFractionDigits: 3 })
const pct = (value: number) => (value * 100).toFixed(1) + '%'
const points = (value: number) => (value >= 0 ? '+' : '−') + (Math.abs(value) * 100).toFixed(2) + ' pp'
const link = (page: Page) => '#' + page.toLowerCase()
const current = () => pages.find(page => link(page) === location.hash) ?? 'Overview'
const education = 'This educational demonstration estimates the dataset’s recorded label. It is not a diagnosis, a clinical screen, or a treatment recommendation.'

function Logo() {
  return <img className="brand-logo" src="/assets/MindScope.png" alt="MindScope" width="2172" height="724"/>
}

function Heading({ label, title, children }: { label: string; title: string; children: ReactNode }) {
  return <header className="heading"><span className="eyebrow">{label}</span><h1>{title}</h1><p>{children}</p></header>
}

function Notice({ children, error = false }: { children: ReactNode; error?: boolean }) {
  return <div className={error ? 'notice error' : 'notice'} role={error ? 'alert' : 'status'}>{children}</div>
}

function Remote<T>({ path, children }: { path: string; children: (data: T) => ReactNode }) {
  const [response, setResponse] = useState<{ data?: T; error?: string }>({})
  const [attempt, retry] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    const timer = window.setTimeout(() => controller.abort(), 90000)
    request<T>(path, undefined, controller.signal)
      .then(data => setResponse({ data }))
      .catch(error => { if (!controller.signal.aborted) setResponse({ error: error.message }) })
    const timeout = window.setTimeout(() => setResponse(value => value.data || value.error ? value : { error: 'The API took too long to respond.' }), 90000)
    return () => { clearTimeout(timer); clearTimeout(timeout); controller.abort() }
  }, [path, attempt])
  if (response.error) return <Notice error>{response.error} <button className="quiet" onClick={() => { setResponse({}); retry(n => n + 1) }}>Retry</button></Notice>
  return response.data === undefined ? <Notice>Loading research data…</Notice> : children(response.data)
}

function Bars({ items, format = number.format }: { items: { name: string; value: number }[]; format?: (value: number) => string }) {
  const max = Math.max(...items.map(item => item.value), Number.EPSILON)
  return <ul className="bars">{items.map(({ name, value }) => <li key={name}><span>{name}</span><div className="track" aria-hidden="true"><i style={{ width: (value / max * 100) + '%' }}/></div><b>{format(value)}</b></li>)}</ul>
}

function Overview() {
  return <>
    <section className="hero">
      <div className="hero-copy"><span className="eyebrow">Student mental health research</span>
        <h1>Inside a student<br/><em>survey prediction.</em></h1>
        <p>Start with a sample student profile. Follow five XGBoost models from survey responses to a label estimate, then explore the features behind it.</p>
        <div className="actions"><a className="button primary" href={link('Assessment')}>Try a sample profile <span aria-hidden="true">→</span></a><a className="button ghost" href={link('Dataset')}>Explore the dataset</a></div>
      </div>
      <figure className="ensemble-figure">
        <figcaption>Inside the ensemble</figcaption>
        <svg viewBox="0 0 320 162" role="img" aria-label="Five model outputs converge into one average estimate">
          {[20,50,81,112,142].map((y,i) => <g key={y}>
            <path d={'M 50 '+y+' C 146 '+y+' 142 81 223 81'} fill="none" stroke={i===2?'#edc277':'#80d8ac'} strokeWidth="1.8"/>
            <circle cx="50" cy={y} r="4" fill={i===2?'#edc277':'#a5f2ce'}/>
            <text x="7" y={y+4}>M{i+1}</text></g>)}
          <circle cx="247" cy="81" r="29" fill="#124f42" stroke="#a5f2ce"/><text x="247" y="86" textAnchor="middle">avg</text>
        </svg>
        <div className="figure-key"><span>Model outputs</span><span>Mean estimate</span></div>
      </figure>
    </section>
    <Remote<Summary> path="/api/dataset/summary">{data => <div className="stat-grid">
      <div><small>Source records</small><strong>{number.format(data.source_rows)}</strong><span>OpenML dataset 46874</span></div>
      <div><small>Model inputs</small><strong>{data.model_feature_count}</strong><span>Academic, lifestyle and demographic features</span></div>
      <div><small>Ensemble members</small><strong>5</strong><span>XGBoost models trained with different seeds</span></div>
    </div>}</Remote>
    <section className="overview-bottom"><article className="panel"><h2>How the estimate is made</h2><p>Each model estimates the probability of the dataset’s Yes label. MindScope averages the five estimates and applies the threshold of 0.56.</p><a href={link('Methodology')}>Read the methodology <span aria-hidden>↗</span></a></article><article className="panel"><h2>How to read the output</h2><p>{education} SHAP describes how the model uses the inputs.</p><a href={link('Explainability')}>Inspect model explanations <span aria-hidden>↗</span></a></article></section>
  </>
}

const exampleNumbers: Record<string, number> = {
  Age: 21, 'Academic Pressure': 3, 'Work Pressure': 0, CGPA: 7.5,
  'Study Satisfaction': 3, 'Job Satisfaction': 0, 'Work/Study Hours': 6, 'Financial Stress': 3,
}
const exampleCategories: Record<string, string> = {
  Gender: 'Female', City: 'Delhi', Profession: 'Student', 'Sleep Duration': '7-8 hours',
  'Dietary Habits': 'Moderate', Degree: 'B.Sc', 'Family History of Mental Illness': 'No',
}

function Assessment({ onResult }: { onResult: (value: Example) => void }) {
  return <><Heading label="Interactive inference" title="Research assessment">Use a hypothetical profile to explore how the model responds. {education}</Heading>
    <Remote<Model> path="/api/model/info">{model => <AssessmentForm model={model} onResult={onResult}/>}</Remote></>
}

function AssessmentForm({ model, onResult }: { model: Model; onResult: (value: Example) => void }) {
  const [values, setValues] = useState<Record<string, string>>({})
  const [result, setResult] = useState<Example>()
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [explanationError, setExplanationError] = useState('')
  const [touched, setTouched] = useState(false)
  const progress = model.feature_columns.filter(name => values[name]?.trim()).length

  function loadExample() {
    const pairs = model.feature_columns.map(name => {
      const options = model.category_options[name]
      const preferred = options?.find(option => option.replace(/['"]/g, '') === exampleCategories[name])
      const bounds = model.numeric_bounds[name]
      return [name, options ? preferred ?? options[0] : String(Math.max(bounds.min, Math.min(bounds.max, exampleNumbers[name])))]
    })
    setValues(Object.fromEntries(pairs))
    setTouched(false)
  }

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const payload = Object.fromEntries(model.feature_columns.map(name => [
      name, model.category_options[name] ? values[name] : Number(values[name]),
    ]))
    setBusy(true); setError(''); setExplanationError(''); setResult(undefined); setTouched(false)
    try {
      const prediction = await request<Prediction>('/api/predict', payload)
      setResult({ prediction }); onResult({ prediction })
      try {
        const explanation = await request<Explanation>('/api/explain', payload)
        setResult({ prediction, explanation }); onResult({ prediction, explanation })
      } catch (failure) {
        setExplanationError((failure as Error).message)
      }
    } catch (failure) {
      setError((failure as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return <div className="assessment-layout"><form className="panel" onSubmit={submit}>
    <div className="panel-title"><div><h2>Profile features</h2><p>{progress} of {model.feature_columns.length} completed · All fields required</p></div><button type="button" className="quiet" disabled={busy} onClick={() => {loadExample(); setTouched(true)}}>Load sample profile</button></div>
    <div className="fields">{model.feature_columns.map((name, index) => {
      const choices = model.category_options[name]
      const bounds = model.numeric_bounds[name]
      const id = 'feature-' + index
      const update = (value: string) => { setValues(previous => ({ ...previous, [name]: value })); setTouched(true) }
      return <div className="field" key={name}><label htmlFor={id}>{name}</label>{choices ?
        <select id={id} value={values[name] ?? ''} onChange={event => update(event.target.value)} required disabled={busy}>
          <option value="">Select an option</option>{choices.map(choice => <option key={choice} value={choice}>{choice}</option>)}
        </select> :
        <input id={id} type="number" step="any" min={bounds.min} max={bounds.max} placeholder={bounds.min + '–' + bounds.max} value={values[name] ?? ''} onChange={event => update(event.target.value)} required disabled={busy}/>}</div>
    })}</div>
    <button className="button primary" disabled={busy}> {busy ? 'Analyzing profile…' : 'Run model'}</button>
    {error && <Notice error>{error}</Notice>}
  </form><aside className="panel result" aria-live="polite"><h2>Model output</h2>
    {result ? <>
      {touched && <Notice>The results below describe the last submitted profile.</Notice>}
      <p className="result-label">Estimated dataset label <strong>{result.prediction.dataset_label_estimate}</strong></p>
      <Bars items={[{ name: 'Yes', value: result.prediction.probability_yes }, { name: 'No', value: result.prediction.probability_no }]} format={pct}/>
      <p>Decision threshold {result.prediction.threshold.toFixed(2)} for Yes.</p>
      {result.explanation && <Waterfall explanation={result.explanation}/>}
      {explanationError && <Notice error>{explanationError}</Notice>}
      {!result.explanation && busy && <Notice>Calculating the explanation…</Notice>}
    </> : <p>{busy ? 'Running the five ensemble members…' : 'Run the model to see its estimated label and feature contributions.'}</p>}
    <p className="caption">{education}</p>
  </aside></div>
}

function Waterfall({ explanation }: { explanation: Explanation }) {
  let cumulative = explanation.base_value
  const steps = explanation.contributions.map(({ feature, contribution }) => {
    const from = cumulative
    cumulative += contribution
    return { feature, contribution, from, to: cumulative }
  })
  const low = Math.min(0, ...steps.map(step => step.to))
  const high = Math.max(1, ...steps.map(step => step.to))
  const x = (value: number) => (value - low) / (high - low) * 100
  return <section className="waterfall"><h3>Prediction waterfall</h3><p>Baseline {pct(explanation.base_value)} <span aria-hidden>→</span> estimate {pct(explanation.model_output)}</p>
    <ul className="bars">{steps.map(({ feature, contribution, from, to }) => <li key={feature}><span>{feature}</span>
      <div className="track" aria-hidden="true"><i className={contribution < 0 ? 'negative' : ''} style={{ left: x(Math.min(from, to)) + '%', width: Math.abs(x(to) - x(from)) + '%' }}/></div><b>{points(contribution)}</b></li>)}</ul>
    <p className="caption">Each bar starts where the previous contribution ends. Values use percentage points.</p>
  </section>
}

function Explainability({ example }: { example?: Example }) {
  return <><Heading label="Explainable AI" title="Inside the ensemble">See which features shift the predicted Yes probability across profiles and within a single example.</Heading>
    <Remote<GlobalShap> path="/api/explain/global">{data => <><section className="panel"><h2>Global feature importance</h2><p>Mean absolute SHAP contribution across {data.sample_size} sampled profiles, in percentage points.</p>
      <Bars items={data.features.map(item => ({ name: item.feature, value: item.mean_absolute_shap }))} format={value => (value * 100).toFixed(2) + ' pp'}/>
      <p className="caption">Maximum reconstruction error {data.maximum_additivity_error.toExponential(2)} on the probability scale.</p></section>
      <section className="panel plot"><h2>SHAP summary</h2><p>Each point shows the contribution of a transformed feature for one sampled profile.</p><img src={apiUrl('/api/explain/global/summary-plot')} alt="SHAP beeswarm plot for the most influential transformed features" loading="lazy"/></section>
      <section className="panel"><h2>Feature dependence</h2><div className="plots">{data.dependence_plots.map(item => <figure key={item.path}><img src={apiUrl(item.path)} alt={'SHAP dependence for ' + item.feature} loading="lazy"/><figcaption>{item.feature}</figcaption></figure>)}</div></section>
    </>}</Remote>
    <section className="panel"><h2>Individual explanation</h2>{example?.explanation ? <Waterfall explanation={example.explanation}/> : <p>Run a <a href={link('Assessment')}>sample assessment</a> to inspect its waterfall here.</p>}</section>
  </>
}

function Dataset() {
  return <><Heading label="OpenML 46874" title="Dataset explorer">Explore descriptive statistics and associations for the student cohort used by the application.</Heading>
    <Remote<Summary> path="/api/dataset/summary">{summary => <div className="overview-bottom">
      <section className="panel"><h2>Cohort & recorded label</h2><p>{number.format(summary.source_rows)} source records · {number.format(summary.model_population_rows)} after filtering and deduplication</p>
        <Bars items={Object.entries(summary.target_counts).map(([name, value]) => ({ name, value }))}/></section>
      <section className="panel"><h2>Missing values</h2><p>Counts in the displayed cohort and public feature set.</p>
        <Bars items={Object.entries(summary.missing_by_column).filter(([, value]) => value > 0).map(([name, value]) => ({ name, value }))}/>
      </section></div>}</Remote>
    <Remote<{ features: Feature[] }> path="/api/analytics/features">{data => <FeatureExplorer features={data.features}/>}</Remote>
    <Remote<Correlations> path="/api/analytics/correlations">{data => <section className="panel"><h2>Numeric correlations</h2><p>Pearson correlation. A dash indicates insufficient variation.</p>
      <div className="table-wrap"><table><thead><tr><th scope="col">Feature</th>{data.features.map(name => <th key={name} scope="col">{name}</th>)}</tr></thead>
      <tbody>{data.features.map(name => <tr key={name}><th scope="row">{name}</th>{data.features.map(other => {
        const value = data.matrix[name][other]
        return <td key={other} style={{ backgroundColor: value == null ? undefined : value < 0 ? 'rgba(180,99,62,' + Math.abs(value) * .27 + ')' : 'rgba(45,148,103,' + value * .27 + ')' }}>{value?.toFixed(2) ?? '—'}</td>
      })}</tr>)}</tbody></table></div></section>}</Remote>
  </>
}

function FeatureExplorer({ features }: { features: Feature[] }) {
  const [name, setName] = useState('Age')
  const feature = features.find(item => item.name === name) ?? features[0]
  return <section className="panel"><div className="panel-title"><div><h2>Feature distributions</h2><p>{number.format(feature.missing_count)} missing</p></div>
    <select aria-label="Choose feature" value={feature.name} onChange={event => setName(event.target.value)}>{features.map(item => <option key={item.name}>{item.name}</option>)}</select></div>
    {feature.statistics && <dl className="stats-inline">{Object.entries(feature.statistics).map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{number.format(value)}</dd></div>)}</dl>}
    <Bars items={feature.distribution.map(({ category, count }) => ({ name: category, value: count }))}/>
  </section>
}

type Metrics = {
  message: string
  held_out?: {
    sample_count: number; threshold: number; accuracy: number; yes_precision: number
    yes_recall: number; yes_f1: number; macro_f1: number; roc_auc: number; pr_auc: number; brier: number
    confusion_matrix: number[][]
    calibration: { mean_prediction: number; observed_yes_rate: number }[]
  }
  notebook_evaluation?: {
    validation_macro_f1: number; test_accuracy: number; test_macro_f1: number; test_roc_auc: number
  }
  verification: { reference_profiles: number; absolute_error: number; maximum_reference_error?: number }
}

function Performance() {
  return <><Heading label="Evaluation evidence" title="Model performance">Held-out results for the recorded dataset label.</Heading>
    <Remote<Metrics> path="/api/model/metrics">{data => <>
      <section className="panel"><h2>Exported ensemble</h2><p>{data.message}</p>
        {data.held_out && <HeldOutResults result={data.held_out}/>}
      </section>
      {data.notebook_evaluation && <section className="panel"><h2>Notebook results before refit</h2>
        <p>The train-only ensemble reached a validation macro-F1 of {pct(data.notebook_evaluation.validation_macro_f1)} at the selected threshold. Its held-out accuracy was {pct(data.notebook_evaluation.test_accuracy)}, with macro-F1 {pct(data.notebook_evaluation.test_macro_f1)} and ROC-AUC {data.notebook_evaluation.test_roc_auc.toFixed(4)}.</p>
      </section>}
      <section className="panel"><h2>Prediction parity</h2><p>The serving model reproduced {data.verification.reference_profiles} saved notebook predictions. Maximum absolute probability difference {(data.verification.maximum_reference_error ?? data.verification.absolute_error).toExponential(2)}.</p></section>
    </>}</Remote></>
}

function HeldOutResults({ result }: { result: NonNullable<Metrics['held_out']> }) {
  const metrics: [string, string][] = [
    ['Accuracy', pct(result.accuracy)], ['Macro-F1', pct(result.macro_f1)],
    ['Yes precision', pct(result.yes_precision)], ['Yes recall', pct(result.yes_recall)],
    ['Yes F1', pct(result.yes_f1)], ['ROC-AUC', result.roc_auc.toFixed(4)],
    ['PR-AUC (average precision)', result.pr_auc.toFixed(4)], ['Brier score', result.brier.toFixed(4)],
  ]
  return <>
    <p>{number.format(result.sample_count)} held-out records · Decision threshold {result.threshold.toFixed(2)}</p>
    <dl className="stats-inline">{metrics.map(([label, value]) =>
      <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>
    <div className="overview-bottom">
      <section><h3>Confusion matrix</h3><div className="table-wrap"><table>
        <thead><tr><th scope="col">Recorded label</th><th scope="col">Predicted No</th><th scope="col">Predicted Yes</th></tr></thead>
        <tbody>{result.confusion_matrix.map((row, index) => <tr key={index}><th scope="row">{index === 0 ? 'No' : 'Yes'}</th>
          {row.map((count, column) => <td key={column}>{number.format(count)}</td>)}</tr>)}</tbody>
      </table></div></section>
      <section><h3>Calibration on the test split</h3>
        <p className="caption">Each point shows a probability bin. The diagonal represents perfect agreement.</p>
        <div className="calibration" role="img" aria-label="Observed Yes rate versus mean predicted Yes probability, from zero to one on both axes. Exact values are in the table below.">
          {result.calibration.map(({ mean_prediction, observed_yes_rate }, index) =>
            <span key={index} style={{ left: mean_prediction * 100 + '%', bottom: observed_yes_rate * 100 + '%' }}/>)}
        </div><p className="caption">Mean predicted Yes probability →</p>
        <details><summary>View bin values</summary><div className="table-wrap"><table>
          <thead><tr><th scope="col">Mean prediction</th><th scope="col">Observed Yes rate</th></tr></thead>
          <tbody>{result.calibration.map(({ mean_prediction, observed_yes_rate }, index) =>
            <tr key={index}><td>{pct(mean_prediction)}</td><td>{pct(observed_yes_rate)}</td></tr>)}</tbody>
        </table></div></details>
      </section>
    </div>
  </>
}

function Methodology() {
  return <><Heading label="Research notes" title="Methodology & responsible use">How the data, model, and explanations are connected.</Heading>
    <div className="methods">{[
      ['Dataset', 'OpenML dataset 46874, version 2, supplies the recorded Yes/No label. The application displays student-only aggregates after full-row deduplication. The source includes a self-harm-related field that is excluded from model inputs and public analytics.'],
      ['Preprocessing', 'Each member applies its saved pipeline, including median imputation, most-frequent categorical imputation, and one-hot encoding. The application takes its allowed categories from the fitted encoders.'],
      ['Ensemble and threshold', 'The ensemble averages the five Yes probabilities from seeds 42 through 46. The exported configuration specifies 0.56 as the decision threshold.'],
      ['SHAP', 'Probability-space interventional TreeSHAP is computed for all five members using a fixed 64-profile background. Averaged contributions are grouped under the original features. The baseline plus contributions reconstructs the estimated Yes probability within a numerical tolerance.'],
      ['Evaluation', 'Notebook predictions check inference parity. The 0.56 threshold was selected on validation data. The final exported ensemble was fit on the combined development set and scored on the original holdout.'],
      ['Use and privacy', education + ' Results describe model behavior for the supplied inputs. The application does not save submitted profiles. Survey coverage, label provenance, and generalization to other student populations are limited.'],
    ].map(([title, description]) => <article className="panel" key={title}><h2>{title}</h2><p>{description}</p></article>)}</div>
    <p><a href="https://www.openml.org/search?type=data&sort=runs&id=46874">View the original OpenML dataset ↗</a></p></>
}

function App() {
  const [page, setPage] = useState(current)
  const [example, setExample] = useState<Example>()
  const main = useRef<HTMLElement>(null)
  useEffect(() => {
    const update = () => { setPage(current()); window.scrollTo(0, 0); main.current?.focus() }
    window.addEventListener('hashchange', update)
    return () => window.removeEventListener('hashchange', update)
  }, [])
  return <><a className="skip" href="#main-content" onClick={event => { event.preventDefault(); main.current?.focus() }}>Skip to content</a>
    <header className="site-header"><div className="nav-inner"><a href={link('Overview')} className="brand"><Logo/></a>
      <nav aria-label="Main">{pages.map(item => <a key={item} href={link(item)} aria-current={page === item ? 'page' : undefined}>{item}</a>)}</nav>
    </div></header>
    <main id="main-content" ref={main} tabIndex={-1}>
      {page === 'Overview' && <Overview/>}
      {page === 'Assessment' && <Assessment onResult={setExample}/>}
      {page === 'Explainability' && <Explainability example={example}/>}
      {page === 'Dataset' && <Dataset/>}
      {page === 'Performance' && <Performance/>}
      {page === 'Methodology' && <Methodology/>}
    </main>
    <footer><div className="footer-inner">
      <div className="footer-project"><a href={link('Overview')} className="brand">MindScope</a><p>Educational research demonstration</p></div>
      <section className="developer" aria-label="Developer profile"><span className="eyebrow">Developed by</span>
        <h2>H M AHSAN HABIB KUSHAL</h2><p>Machine Learning Engineer</p><p className="affiliation">Computer Science &amp; Engineering · NITER, University of Dhaka</p>
        <div className="profile-links"><a href="https://ahsanhabibkushal.vercel.app/">Portfolio ↗</a><a href="https://github.com/AhsanCS64">GitHub ↗</a></div>
      </section>
    </div></footer>
  </>
}

createRoot(document.getElementById('root')!).render(<App/>)
