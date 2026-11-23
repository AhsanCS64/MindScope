export type Model = {
  model_version: string
  threshold: number
  feature_columns: string[]
  category_options: Record<string, string[]>
  numeric_bounds: Record<string, { min: number; max: number }>
  disclaimer: string
}
export type Summary = {
  source_rows: number
  model_population_rows: number
  model_feature_count: number
  target_counts: Record<string, number>
  missing_by_column: Record<string, number>
}
export type Feature = {
  name: string
  kind: string
  missing_count: number
  statistics?: Record<string, number>
  distribution: { category: string; count: number }[]
}
export type Correlations = { features: string[]; matrix: Record<string, Record<string, number | null>> }
export type GlobalShap = {
  sample_size: number
  maximum_additivity_error: number
  features: { feature: string; mean_absolute_shap: number }[]
  dependence_plots: { feature: string; path: string }[]
}
export type Prediction = {
  probability_yes: number
  probability_no: number
  dataset_label_estimate: string
  threshold: number
  model_version: string
}
export type Explanation = {
  base_value: number
  model_output: number
  additivity_error: number
  contributions: { feature: string; contribution: number }[]
}

const base = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
export const apiUrl = (path: string) => base + path

export async function request<T>(path: string, body?: Record<string, string | number>, signal?: AbortSignal): Promise<T> {
  const response = await fetch(apiUrl(path), {
    signal: signal ?? AbortSignal.timeout(90_000),
    ...(body && { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }),
  }).catch(() => { throw new Error('The API could not be reached. Check the connection and try again.') })
  if (!response.ok) {
    const data = await response.json().catch(() => null)
    const detail = typeof data?.detail === 'string' ? data.detail : data?.detail?.message
    throw new Error(detail ?? (response.status === 422 ? 'Review the supplied feature values.' : 'The API could not complete this request. Try again.'))
  }
  return response.json() as Promise<T>
}
